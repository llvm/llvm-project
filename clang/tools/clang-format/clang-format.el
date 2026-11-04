;;; clang-format.el --- Format code using clang-format  -*- lexical-binding: t; -*-

;; Version: 0.1.0
;; Keywords: tools, c
;; Package-Requires: ((cl-lib "0.3"))
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

;;; Commentary:

;; This package allows to filter code through clang-format to fix its formatting.
;; clang-format is a tool that formats C/C++/Obj-C code according to a set of
;; style options, see <http://clang.llvm.org/docs/ClangFormatStyleOptions.html>.
;; Note that clang-format 3.4 or newer is required.

;; clang-format.el is available via MELPA and can be installed via
;;
;;   M-x package-install clang-format
;;
;; when ("melpa" . "http://melpa.org/packages/") is included in
;; `package-archives'.  Alternatively, ensure the directory of this
;; file is in your `load-path' and add
;;
;;   (require 'clang-format)
;;
;; to your .emacs configuration.

;; You may also want to bind `clang-format-region' to a key:
;;
;;   (global-set-key [C-M-tab] 'clang-format-region)

;;; Code:

(require 'cl-lib)
(require 'xml)
(require 'vc-git)

(defgroup clang-format nil
  "Format code using clang-format."
  :group 'tools)

(defcustom clang-format-executable
  (or (executable-find "clang-format")
      "clang-format")
  "Location of the clang-format executable.

A string containing the name or the full path of the executable."
  :group 'clang-format
  :type '(file :must-match t)
  :risky t)

(defcustom clang-format-style nil
  "Style argument to pass to clang-format.

By default clang-format will load the style configuration from
a file named .clang-format located in one of the parent directories
of the buffer."
  :group 'clang-format
  :type '(choice (string) (const nil))
  :safe #'stringp)
(make-variable-buffer-local 'clang-format-style)

(defcustom clang-format-fallback-style "none"
  "Fallback style to pass to clang-format.

This style will be used if clang-format-style is set to \"file\"
and no .clang-format is found in the directory of the buffer or
one of parent directories. Set to \"none\" to disable formatting
in such buffers."
  :group 'clang-format
  :type 'string
  :safe #'stringp)
(make-variable-buffer-local 'clang-format-fallback-style)

(defcustom clang-format-on-save-p 'clang-format-on-save-check-config-exists
  "Only reformat on save if this function returns non-nil.

You may wish to choose one of the following options:
- `always': To always format on save.
- `clang-format-on-save-check-config-exists':
  Only reformat when \".clang-format\" exists.

Otherwise you can set this to a user defined function."
  :group 'clang-format
  :type 'function
  :risky t)
(make-variable-buffer-local 'clang-format-on-save-p)

(defun clang-format--extract (xml-node)
  "Extract replacements and cursor information from XML-NODE."
  (unless (and (listp xml-node) (eq (xml-node-name xml-node) 'replacements))
    (error "Expected <replacements> node"))
  (let ((nodes (xml-node-children xml-node))
        (incomplete-format (xml-get-attribute xml-node 'incomplete_format))
        replacements
        cursor)
    (dolist (node nodes)
      (when (listp node)
        (let* ((children (xml-node-children node))
               (text (car children)))
          (cl-case (xml-node-name node)
            (replacement
             (let* ((offset (xml-get-attribute-or-nil node 'offset))
                    (length (xml-get-attribute-or-nil node 'length)))
               (when (or (null offset) (null length))
                 (error "<replacement> node does not have offset and length attributes"))
               (when (cdr children)
                 (error "More than one child node in <replacement> node"))

               (setq offset (string-to-number offset))
               (setq length (string-to-number length))
               (push (list offset length text) replacements)))
            (cursor
             (setq cursor (string-to-number text)))))))

    ;; Sort by decreasing offset, length.
    (setq replacements (sort (delq nil replacements)
                             (lambda (a b)
                               (or (> (car a) (car b))
                                   (and (= (car a) (car b))
                                        (> (cadr a) (cadr b)))))))

    (list replacements cursor (string= incomplete-format "true"))))

(defun clang-format--replace (offset length &optional text)
  "Replace the region defined by OFFSET and LENGTH with TEXT.
OFFSET and LENGTH are measured in bytes, not characters.  OFFSET
is a zero-based file offset, assuming ‘utf-8-unix’ coding."
  (let ((start (clang-format--filepos-to-bufferpos offset 'exact 'utf-8-unix))
        (end (clang-format--filepos-to-bufferpos (+ offset length) 'exact
                                                 'utf-8-unix)))
    (goto-char start)
    (delete-region start end)
    (when text
      (insert text))))

;; ‘bufferpos-to-filepos’ and ‘filepos-to-bufferpos’ are new in Emacs 25.1.
;; Provide fallbacks for older versions.
(defalias 'clang-format--bufferpos-to-filepos
  (if (fboundp 'bufferpos-to-filepos)
      'bufferpos-to-filepos
    (lambda (position &optional _quality _coding-system)
      (1- (position-bytes position)))))

(defalias 'clang-format--filepos-to-bufferpos
  (if (fboundp 'filepos-to-bufferpos)
      'filepos-to-bufferpos
    (lambda (byte &optional _quality _coding-system)
      (byte-to-position (1+ byte)))))

(defmacro clang-format--with-delete-files-guard (bind-files-to-delete &rest body)
  "Execute BODY which may add temp files to BIND-FILES-TO-DELETE."
  (declare (indent 1))
  `(let ((,bind-files-to-delete nil))
     (unwind-protect
         (progn
           ,@body)
       (while ,bind-files-to-delete
         (with-demoted-errors "failed to remove file: %S"
           (delete-file (pop ,bind-files-to-delete)))))))


(defun clang-format--vc-diff-get-diff-lines (file-orig file-new)
  "Return all line regions that contain diffs between FILE-ORIG and
FILE-NEW.  If there is no diff ‘nil’ is returned. Otherwise the return
is a ‘list’ of line ranges to format. The list of line ranges can be
passed to ‘clang-format--region-impl’"
  ;; Use temporary buffer for output of diff.
  (with-temp-buffer
    ;; We could use diff.el:diff-no-select here. The reason we don't
    ;; is diff-no-select requires extra copies on the buffers which
    ;; induces noticeable slowdowns, especially on larger files.
    (let ((status (call-process
                   diff-command
                   nil
                   (current-buffer)
                   nil
                   ;; Binary diff has different behaviors that we
                   ;; aren't interested in.
                   "-a"
                   ;; Get minimal diff (copy diff config for git-clang-format).
                   "-U0"
                   file-orig
                   file-new))
          (stderr (concat (if (zerop (buffer-size)) "" ": ")
                          (buffer-substring-no-properties
                           (point-min) (line-end-position))))
          (diff-lines '()))
      (cond
       ((stringp status)
        (error "clang-format: (diff killed by signal %s%s)" status stderr))
       ;; Return of 0 indicates no diff.
       ((= status 0) nil)
       ;; Return of 1 indicates found diffs and no error.
       ((= status 1)
        ;; Find and collect all diff lines.
        ;; We are matching something like:
        ;; "@@ -80 +80 @@" or "@@ -80,2 +80,2 @@"
        (goto-char (point-min))
        (while (re-search-forward
                "^@@[[:blank:]]-[[:digit:],]+[[:blank:]]\\+\\([[:digit:]]+\\)\\(,\\([[:digit:]]+\\)\\)?[[:blank:]]@@$"
                nil
                t
                1)
          (let ((match1 (string-to-number (match-string 1)))
                (match3 (let ((match3_or_nil (match-string 3)))
                          (if match3_or_nil
                              (string-to-number match3_or_nil)
                            nil))))
            (push (cons match1 (if match3 (+ match1 match3) match1)) diff-lines)))
        (nreverse diff-lines))
       ;; Any return != 0 && != 1 indicates some level of error.
       (t
        (error "clang-format: (diff returned unsuccessfully %s%s)" status stderr))))))

(defun clang-format--vc-diff-get-vc-head-file (tmpfile-vc-head)
  "Stores the contents of ‘buffer-file-name’ at vc revision HEAD into
‘tmpfile-vc-head’. If the current buffer is either not a file or not
in a vc repo, this results in an error. Currently git is the only
supported vc."
  ;; We need the current buffer to be a file.
  (unless (buffer-file-name)
    (error "clang-format: Buffer is not visiting a file"))

  (let ((base-dir (vc-root-dir))
        (backend (vc-backend (buffer-file-name))))
    ;; We need to be able to find version control (git) root.
    (unless base-dir
      (error "clang-format: File not known to git"))
    (cond
     ((string-equal backend "Git")
      ;; Get the filename relative to git root.
      (let ((vc-file-name (substring
                           (expand-file-name (buffer-file-name))
                           (string-width (expand-file-name base-dir))
                           nil)))
        (let ((status (call-process
                       vc-git-program
                       nil
                       `(:file ,tmpfile-vc-head)
                       nil
                       "show" (concat "HEAD:" vc-file-name)))
              (stderr (with-temp-buffer
                        (unless (zerop (cadr (insert-file-contents tmpfile-vc-head)))
                          (insert ": "))
                        (buffer-substring-no-properties
                         (point-min) (line-end-position)))))
          (when (stringp status)
            (error "clang-format: (git show HEAD:%s killed by signal %s%s)"
                   vc-file-name status stderr))
          (unless (zerop status)
            (error "clang-format: (git show HEAD:%s returned unsuccessfully %s%s)"
                   vc-file-name status stderr)))))
     (t
      (error
       "Version control %s isn't supported, currently supported backends: git"
       backend)))))


(defun clang-format--region-impl (start end &optional style assume-file-name lines)
  "Common implementation for ‘clang-format-buffer’,
‘clang-format-region’, and ‘clang-format-vc-diff’. START and END
refer to the region to be formatter. STYLE and ASSUME-FILE-NAME are
used for configuring the clang-format. And LINES is used to pass
specific locations for reformatting (i.e diff locations)."
  (unless style
    (setq style clang-format-style))

  (unless assume-file-name
    (setq assume-file-name (buffer-file-name (buffer-base-buffer))))

  ;; Convert list of line ranges to list command for ‘clang-format’ executable.
  (when lines
    (setq lines (mapcar (lambda (range)
                          (format "--lines=%d:%d" (car range) (cdr range)))
                        lines)))

  (let ((file-start (clang-format--bufferpos-to-filepos start 'approximate
                                                        'utf-8-unix))
        (file-end (clang-format--bufferpos-to-filepos end 'approximate
                                                      'utf-8-unix))
        (cursor (clang-format--bufferpos-to-filepos (point) 'exact 'utf-8-unix))
        (temp-buffer (generate-new-buffer " *clang-format-temp*"))
        (temp-file (make-temp-file "clang-format"))
        ;; Output is XML, which is always UTF-8.  Input encoding should match
        ;; the encoding used to convert between buffer and file positions,
        ;; otherwise the offsets calculated above are off.  For simplicity, we
        ;; always use ‘utf-8-unix’ and ignore the buffer coding system.
        (default-process-coding-system '(utf-8-unix . utf-8-unix)))
    (unwind-protect
        (let ((status (apply #'call-process-region
                             nil nil clang-format-executable
                             nil `(,temp-buffer ,temp-file) nil
                             `("--output-replacements-xml"
                               ;; Guard against a nil assume-file-name.
                               ;; If the clang-format option -assume-filename
                               ;; is given a blank string it will crash as per
                               ;; the following bug report
                               ;; https://bugs.llvm.org/show_bug.cgi?id=34667
                               ,@(and assume-file-name
                                      (list "--assume-filename" assume-file-name))
                               ,@(and style (list "--style" style))
                               "--fallback-style" ,clang-format-fallback-style
                               ,@(and lines lines)
                               ,@(and (not lines)
                                      (list
                                       "--offset" (number-to-string file-start)
                                       "--length" (number-to-string
                                                   (- file-end file-start))))
                               "--cursor" ,(number-to-string cursor))))
              (stderr (with-temp-buffer
                        (unless (zerop (cadr (insert-file-contents temp-file)))
                          (insert ": "))
                        (buffer-substring-no-properties
                         (point-min) (line-end-position)))))
          (cond
           ((stringp status)
            (error "(clang-format killed by signal %s%s)" status stderr))
           ((not (zerop status))
            (error "(clang-format failed with code %d%s)" status stderr)))

          (cl-destructuring-bind (replacements cursor incomplete-format)
              (with-current-buffer temp-buffer
                (clang-format--extract (car (xml-parse-region))))
            (save-excursion
              (dolist (rpl replacements)
                (apply #'clang-format--replace rpl)))
            (when cursor
              (goto-char (clang-format--filepos-to-bufferpos cursor 'exact
                                                             'utf-8-unix)))
            (if incomplete-format
                (message "(clang-format: incomplete (syntax errors)%s)" stderr)
              (message "(clang-format: success%s)" stderr))))
      (with-demoted-errors
          "clang-format: Failed to delete temporary file: %S"
        (delete-file temp-file))
      (when (buffer-name temp-buffer) (kill-buffer temp-buffer)))))


;;;###autoload
(defun clang-format-vc-diff (&optional style assume-file-name)
  "The same as ‘clang-format-buffer’ but only operates on the vc
diffs from HEAD in the buffer. If no STYLE is given uses
‘clang-format-style’. Use ASSUME-FILE-NAME to locate a style config
file. If no ASSUME-FILE-NAME is given uses the function
‘buffer-file-name’."
  (interactive)
  (clang-format--with-delete-files-guard tmp-files
    (let ((tmpfile-vc-head nil)
          (tmpfile-curbuf nil))
      (setq tmpfile-vc-head
            (make-temp-file "clang-format-vc-tmp-head-content"))
      (push tmpfile-vc-head tmp-files)
      (clang-format--vc-diff-get-vc-head-file tmpfile-vc-head)
      ;; Move the current buffer to a temporary file to take a
      ;; diff. Even if current-buffer is backed by a file, we
      ;; want to diff the buffer contents which might not be
      ;; saved.
      (setq tmpfile-curbuf (make-temp-file "clang-format-vc-tmp"))
      (push tmpfile-curbuf tmp-files)
      (write-region nil nil tmpfile-curbuf nil 'nomessage)
      ;; Get a list of lines with a diff.
      (let ((diff-lines
             (clang-format--vc-diff-get-diff-lines
              tmpfile-vc-head tmpfile-curbuf)))
        ;; If we have any diffs, format them.
        (when diff-lines
          (clang-format--region-impl
           (point-min)
           (point-max)
           style
           assume-file-name
           diff-lines))))))


;;;###autoload
(defun clang-format-region (start end &optional style assume-file-name)
  "Use clang-format to format the code between START and END according
to STYLE.  If called interactively uses the region or the current
statement if there is no no active region. If no STYLE is given uses
`clang-format-style'. Use ASSUME-FILE-NAME to locate a style config
file, if no ASSUME-FILE-NAME is given uses the function
`buffer-file-name'."
  (interactive
   (if (use-region-p)
       (list (region-beginning) (region-end))
     (list (point) (point))))
  (clang-format--region-impl start end style assume-file-name))

;;;###autoload
(defun clang-format-buffer (&optional style assume-file-name)
  "Use clang-format to format the current buffer according to STYLE.
If no STYLE is given uses `clang-format-style'. Use ASSUME-FILE-NAME
to locate a style config file. If no ASSUME-FILE-NAME is given uses
the function `buffer-file-name'."
  (interactive)
  (clang-format--region-impl
   (point-min)
   (point-max)
   style
   assume-file-name))

;;;###autoload
(defalias 'clang-format 'clang-format-region)

;; Format on save minor mode.

(defun clang-format--on-save-buffer-hook ()
  "The hook to run on buffer saving to format the buffer."
  ;; Demote errors as this is user configurable, we can't be sure it wont error.
  (when (with-demoted-errors "clang-format: Error %S"
          (funcall clang-format-on-save-p))
    (clang-format-buffer))
  ;; Continue to save.
  nil)

(defun clang-format--on-save-enable ()
  "Disable the minor mode."
  (add-hook 'before-save-hook #'clang-format--on-save-buffer-hook nil t))

(defun clang-format--on-save-disable ()
  "Enable the minor mode."
  (remove-hook 'before-save-hook #'clang-format--on-save-buffer-hook t))

;; Default value for `clang-format-on-save-p'.
(defun clang-format-on-save-check-config-exists ()
  "Return non-nil when `.clang-format' is found in a parent directory."
  ;; Unlikely but possible this is nil.
  (let ((filepath buffer-file-name))
    (cond
     (filepath
      (not (null (locate-dominating-file (file-name-directory filepath) ".clang-format"))))
     (t
      nil))))

;;;###autoload
(define-minor-mode clang-format-on-save-mode
  "Clang-format on save minor mode."
  :global nil
  :lighter ""
  :keymap nil

  (cond
   (clang-format-on-save-mode
    (clang-format--on-save-enable))
   (t
    (clang-format--on-save-disable))))

(provide 'clang-format)
;;; clang-format.el ends here
