;;; clang-format.el --- Format code using clang-format

;; Keywords: tools, c
;; Package-Requires: ((cl-lib "0.3"))

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
;; `package-archives'. Alternatively, ensure the directory of this
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

(defgroup clang-format nil
  "Format code using clang-format."
  :group 'tools)

(defcustom clang-format-executable
  (or (executable-find "clang-format")
      "clang-format")
  "Location of the clang-format executable.

A string containing the name or the full path of the executable."
  :group 'clang-format
  :type 'string
  :risky t)

(defcustom clang-format-style "file"
  "Style argument to pass to clang-format.

By default clang-format will load the style configuration from
a file named .clang-format located in one of the parent directories
of the buffer."
  :group 'clang-format
  :type 'string
  :safe #'stringp)
(make-variable-buffer-local 'clang-format-style)

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
            ('replacement
             (let* ((offset (xml-get-attribute-or-nil node 'offset))
                    (length (xml-get-attribute-or-nil node 'length)))
               (when (or (null offset) (null length))
                 (error "<replacement> node does not have offset and length attributes"))
               (when (cdr children)
                 (error "More than one child node in <replacement> node"))

               (setq offset (string-to-number offset))
               (setq length (string-to-number length))
               (push (list offset length text) replacements)))
            ('cursor
             (setq cursor (string-to-number text)))))))

    ;; Sort by decreasing offset, length.
    (setq replacements (sort (delq nil replacements)
                             (lambda (a b)
                               (or (> (car a) (car b))
                                   (and (= (car a) (car b))
                                        (> (cadr a) (cadr b)))))))

    (list replacements cursor (string= incomplete-format "true"))))

(defun clang-format--replace (offset length &optional text)
  (let ((start (byte-to-position (1+ offset)))
        (end (byte-to-position (+ 1 offset length))))
    (goto-char start)
    (delete-region start end)
    (when text
      (insert text))))

;;;###autoload
(defun clang-format-region (char-start char-end &optional style)
  "Use clang-format to format the code between START and END according to STYLE.
If called interactively uses the region or the current statement if there
is no active region.  If no style is given uses `clang-format-style'."
  (interactive
   (if (use-region-p)
       (list (region-beginning) (region-end))
     (list (point) (point))))

  (unless style
    (setq style clang-format-style))

  (let ((start (1- (position-bytes char-start)))
        (end (1- (position-bytes char-end)))
        (cursor (1- (position-bytes (point))))
        (temp-buffer (generate-new-buffer " *clang-format-temp*"))
        (temp-file (make-temp-file "clang-format")))
    (unwind-protect
        (let (status stderr operations)
          (setq status
                (call-process-region
                 (point-min) (point-max) clang-format-executable
                 nil `(,temp-buffer ,temp-file) nil

                 "-output-replacements-xml"
                 "-sort-includes"
                 "-assume-filename" (or (buffer-file-name) "")
                 "-style" style
                 "-offset" (number-to-string start)
                 "-length" (number-to-string (- end start))
                 "-cursor" (number-to-string cursor)))
          (setq stderr
                (with-temp-buffer
                  (insert-file-contents temp-file)
                  (when (> (point-max) (point-min))
                    (insert ": "))
                  (buffer-substring-no-properties
                   (point-min) (line-end-position))))

          (cond
           ((stringp status)
            (error "(clang-format killed by signal %s%s)" status stderr))
           ((not (equal 0 status))
            (error "(clang-format failed with code %d%s)" status stderr)))

          (with-current-buffer temp-buffer
            (setq operations (clang-format--extract (car (xml-parse-region)))))

          (let ((replacements (nth 0 operations))
                (cursor (nth 1 operations))
                (incomplete-format (nth 2 operations)))
            (save-excursion
              (mapc (lambda (rpl)
                      (apply #'clang-format--replace rpl))
                    replacements))
            (when cursor
              (goto-char (byte-to-position (1+ cursor))))
            (message "%s" incomplete-format)
            (if incomplete-format
                (message "(clang-format: incomplete (syntax errors)%s)" stderr)
              (message "(clang-format: success%s)" stderr))))
      (delete-file temp-file)
      (when (buffer-name temp-buffer) (kill-buffer temp-buffer)))))

;;;###autoload
(defun clang-format-buffer (&optional style)
  "Use clang-format to format the current buffer according to STYLE."
  (interactive)
  (clang-format-region (point-min) (point-max) style))

;;;###autoload
(defalias 'clang-format 'clang-format-region)

(provide 'clang-format)
;;; clang-format.el ends here
