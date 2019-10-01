;;; clang-include-fixer.el --- Emacs integration of the clang include fixer  -*- lexical-binding: t; -*-

;; Keywords: tools, c
;; Package-Requires: ((cl-lib "0.5") (json "1.2") (let-alist "1.0.4"))

;;; Commentary:

;; This package allows Emacs users to invoke the 'clang-include-fixer' within
;; Emacs.  'clang-include-fixer' provides an automated way of adding #include
;; directives for missing symbols in one translation unit, see
;; <http://clang.llvm.org/extra/clang-include-fixer.html>.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'let-alist)

(defgroup clang-include-fixer nil
  "Clang-based include fixer."
  :group 'tools)

(defvar clang-include-fixer-add-include-hook nil
  "A hook that will be called for every added include.
The first argument is the filename of the include, the second argument is
non-nil if the include is a system-header.")

(defcustom clang-include-fixer-executable
  "clang-include-fixer"
  "Location of the clang-include-fixer executable.

A string containing the name or the full path of the executable."
  :group 'clang-include-fixer
  :type '(file :must-match t)
  :risky t)

(defcustom clang-include-fixer-input-format
  'yaml
  "Input format for clang-include-fixer.
This string is passed as -db argument to
`clang-include-fixer-executable'."
  :group 'clang-include-fixer
  :type '(radio
          (const :tag "Hard-coded mapping" :fixed)
          (const :tag "YAML" yaml)
          (symbol :tag "Other"))
  :risky t)

(defcustom clang-include-fixer-init-string
  ""
  "Database initialization string for clang-include-fixer.
This string is passed as -input argument to
`clang-include-fixer-executable'."
  :group 'clang-include-fixer
  :type 'string
  :risky t)

(defface clang-include-fixer-highlight '((t :background "green"))
  "Used for highlighting the symbol for which a header file is being added.")

;;;###autoload
(defun clang-include-fixer ()
  "Invoke the Include Fixer to insert missing C++ headers."
  (interactive)
  (message (concat "Calling the include fixer. "
                   "This might take some seconds. Please wait."))
  (clang-include-fixer--start #'clang-include-fixer--add-header
                              "-output-headers"))

;;;###autoload
(defun clang-include-fixer-at-point ()
  "Invoke the Clang include fixer for the symbol at point."
  (interactive)
  (let ((symbol (clang-include-fixer--symbol-at-point)))
    (unless symbol
      (user-error "No symbol at current location"))
    (clang-include-fixer-from-symbol symbol)))

;;;###autoload
(defun clang-include-fixer-from-symbol (symbol)
  "Invoke the Clang include fixer for the SYMBOL.
When called interactively, prompts the user for a symbol."
  (interactive
   (list (read-string "Symbol: " (clang-include-fixer--symbol-at-point))))
  (clang-include-fixer--start #'clang-include-fixer--add-header
                              (format "-query-symbol=%s" symbol)))

(defun clang-include-fixer--start (callback &rest args)
  "Asynchronously start clang-include-fixer with parameters ARGS.
The current file name is passed after ARGS as last argument.  If
the call was successful the returned result is stored in a
temporary buffer, and CALLBACK is called with the temporary
buffer as only argument."
  (unless buffer-file-name
    (user-error "clang-include-fixer works only in buffers that visit a file"))
  (let ((process (if (and (fboundp 'make-process)
                          ;; ‘make-process’ doesn’t support remote files
                          ;; (https://debbugs.gnu.org/cgi/bugreport.cgi?bug=28691).
                          (not (find-file-name-handler default-directory
                                                       'start-file-process)))
                     ;; Prefer using ‘make-process’ if possible, because
                     ;; ‘start-process’ doesn’t allow us to separate the
                     ;; standard error from the output.
                     (clang-include-fixer--make-process callback args)
                   (clang-include-fixer--start-process callback args))))
    (save-restriction
      (widen)
      (process-send-region process (point-min) (point-max)))
    (process-send-eof process))
  nil)

(defun clang-include-fixer--make-process (callback args)
  "Start a new clang-incude-fixer process using `make-process'.
CALLBACK is called after the process finishes successfully; it is
called with a single argument, the buffer where standard output
has been inserted.  ARGS is a list of additional command line
arguments.  Return the new process object."
  (let ((stdin (current-buffer))
        (stdout (generate-new-buffer "*clang-include-fixer output*"))
        (stderr (generate-new-buffer "*clang-include-fixer errors*")))
    (make-process :name "clang-include-fixer"
                  :buffer stdout
                  :command (clang-include-fixer--command args)
                  :coding 'utf-8-unix
                  :noquery t
                  :connection-type 'pipe
                  :sentinel (clang-include-fixer--sentinel stdin stdout stderr
                                                           callback)
                  :stderr stderr)))

(defun clang-include-fixer--start-process (callback args)
  "Start a new clang-incude-fixer process using `start-file-process'.
CALLBACK is called after the process finishes successfully; it is
called with a single argument, the buffer where standard output
has been inserted.  ARGS is a list of additional command line
arguments.  Return the new process object."
  (let* ((stdin (current-buffer))
         (stdout (generate-new-buffer "*clang-include-fixer output*"))
         (process-connection-type nil)
         (process (apply #'start-file-process "clang-include-fixer" stdout
                         (clang-include-fixer--command args))))
    (set-process-coding-system process 'utf-8-unix 'utf-8-unix)
    (set-process-query-on-exit-flag process nil)
    (set-process-sentinel process
                          (clang-include-fixer--sentinel stdin stdout nil
                                                         callback))
    process))

(defun clang-include-fixer--command (args)
  "Return the clang-include-fixer command line.
Returns a list; the first element is the binary to
execute (`clang-include-fixer-executable'), and the remaining
elements are the command line arguments.  Adds proper arguments
for `clang-include-fixer-input-format' and
`clang-include-fixer-init-string'.  Appends the current buffer's
file name; prepends ARGS directly in front of it."
  (cl-check-type args list)
  `(,clang-include-fixer-executable
    ,(format "-db=%s" clang-include-fixer-input-format)
    ,(format "-input=%s" clang-include-fixer-init-string)
    "-stdin"
    ,@args
    ,(clang-include-fixer--file-local-name buffer-file-name)))

(defun clang-include-fixer--sentinel (stdin stdout stderr callback)
  "Return a process sentinel for clang-include-fixer processes.
STDIN, STDOUT, and STDERR are buffers for the standard streams;
only STDERR may be nil.  CALLBACK is called in the case of
success; it is called with a single argument, STDOUT.  On
failure, a buffer containing the error output is displayed."
  (cl-check-type stdin buffer-live)
  (cl-check-type stdout buffer-live)
  (cl-check-type stderr (or null buffer-live))
  (cl-check-type callback function)
  (lambda (process event)
    (cl-check-type process process)
    (cl-check-type event string)
    (unwind-protect
        (if (string-equal event "finished\n")
            (progn
              (when stderr (kill-buffer stderr))
              (with-current-buffer stdin
                (funcall callback stdout))
              (kill-buffer stdout))
          (when stderr (kill-buffer stdout))
          (message "clang-include-fixer failed")
          (with-current-buffer (or stderr stdout)
            (insert "\nProcess " (process-name process)
                    ?\s event))
          (display-buffer (or stderr stdout))))
    nil))

(defun clang-include-fixer--replace-buffer (stdout)
  "Replace current buffer by content of STDOUT."
  (cl-check-type stdout buffer-live)
  (barf-if-buffer-read-only)
  (cond ((fboundp 'replace-buffer-contents) (replace-buffer-contents stdout))
        ((clang-include-fixer--insert-line stdout (current-buffer)))
        (t (erase-buffer) (insert-buffer-substring stdout)))
  (message "Fix applied")
  nil)

(defun clang-include-fixer--insert-line (from to)
  "Insert a single missing line from the buffer FROM into TO.
FROM and TO must be buffers.  If the contents of FROM and TO are
equal, do nothing and return non-nil.  If FROM contains a single
line missing from TO, insert that line into TO so that the buffer
contents are equal and return non-nil.  Otherwise, do nothing and
return nil.  Buffer restrictions are ignored."
  (cl-check-type from buffer-live)
  (cl-check-type to buffer-live)
  (with-current-buffer from
    (save-excursion
      (save-restriction
        (widen)
        (with-current-buffer to
          (save-excursion
            (save-restriction
              (widen)
              ;; Search for the first buffer difference.
              (let ((chars (abs (compare-buffer-substrings to nil nil from nil nil))))
                (if (zerop chars)
                    ;; Buffer contents are equal, nothing to do.
                    t
                  (goto-char chars)
                  ;; We might have ended up in the middle of a line if the
                  ;; current line partially matches.  In this case we would
                  ;; have to insert more than a line.  Move to the beginning of
                  ;; the line to avoid this situation.
                  (beginning-of-line)
                  (with-current-buffer from
                    (goto-char chars)
                    (beginning-of-line)
                    (let ((from-begin (point))
                          (from-end (progn (forward-line) (point)))
                          (to-point (with-current-buffer to (point))))
                      ;; Search for another buffer difference after the line in
                      ;; question.  If there is none, we can proceed.
                      (when (zerop (compare-buffer-substrings from from-end nil
                                                              to to-point nil))
                        (with-current-buffer to
                          (insert-buffer-substring from from-begin from-end))
                        t))))))))))))

(defun clang-include-fixer--add-header (stdout)
  "Analyse the result of clang-include-fixer stored in STDOUT.
Add a missing header if there is any.  If there are multiple
possible headers the user can select one of them to be included.
Temporarily highlight the affected symbols.  Asynchronously call
clang-include-fixer to insert the selected header."
  (cl-check-type stdout buffer-live)
  (let ((context (clang-include-fixer--parse-json stdout)))
    (let-alist context
      (cond
       ((null .QuerySymbolInfos)
        (message "The file is fine, no need to add a header."))
       ((null .HeaderInfos)
        (message "Couldn't find header for '%s'"
                 (let-alist (car .QuerySymbolInfos) .RawIdentifier)))
       (t
        ;; Users may C-g in prompts, make sure the process sentinel
        ;; behaves correctly.
        (with-local-quit
          ;; Replace the HeaderInfos list by a single header selected by
          ;; the user.
          (clang-include-fixer--select-header context)
          ;; Call clang-include-fixer again to insert the selected header.
          (clang-include-fixer--start
           (let ((old-tick (buffer-chars-modified-tick)))
             (lambda (stdout)
               (when (/= old-tick (buffer-chars-modified-tick))
                 ;; Replacing the buffer now would undo the user’s changes.
                 (user-error (concat "The buffer has been changed "
                                     "before the header could be inserted")))
               (clang-include-fixer--replace-buffer stdout)
               (let-alist context
                 (let-alist (car .HeaderInfos)
                   (with-local-quit
                     (run-hook-with-args 'clang-include-fixer-add-include-hook
                                         (substring .Header 1 -1)
                                         (string= (substring .Header 0 1) "<")))))))
           (format "-insert-header=%s"
                   (clang-include-fixer--encode-json context))))))))
  nil)

(defun clang-include-fixer--select-header (context)
  "Prompt the user for a header if necessary.
CONTEXT must be a clang-include-fixer context object in
association list format.  If it contains more than one HeaderInfo
element, prompt the user to select one of the headers.  CONTEXT
is modified to include only the selected element."
  (cl-check-type context cons)
  (let-alist context
    (if (cdr .HeaderInfos)
        (clang-include-fixer--prompt-for-header context)
      (message "Only one include is missing: %s"
               (let-alist (car .HeaderInfos) .Header))))
  nil)

(defvar clang-include-fixer--history nil
  "History for `clang-include-fixer--prompt-for-header'.")

(defun clang-include-fixer--prompt-for-header (context)
  "Prompt the user for a single header.
The choices are taken from the HeaderInfo elements in CONTEXT.
They are replaced by the single element selected by the user."
  (let-alist context
    (let ((symbol (clang-include-fixer--symbol-name .QuerySymbolInfos))
          ;; Add temporary highlighting so that the user knows which
          ;; symbols the current session is about.
          (overlays (remove nil
                            (mapcar #'clang-include-fixer--highlight .QuerySymbolInfos))))
      (unwind-protect
          (save-excursion
            ;; While prompting, go to the closest overlay so that the user sees
            ;; some context.
            (when overlays
              (goto-char (clang-include-fixer--closest-overlay overlays)))
            (cl-flet ((header (info) (let-alist info .Header)))
              ;; The header-infos is already sorted by clang-include-fixer.
              (let* ((headers (mapcar #'header .HeaderInfos))
                     (header (completing-read
                              (clang-include-fixer--format-message
                               "Select include for '%s': " symbol)
                              headers nil :require-match nil
                              'clang-include-fixer--history
                              ;; Specify a default to prevent the behavior
                              ;; described in
                              ;; https://github.com/DarwinAwardWinner/ido-completing-read-plus#why-does-ret-sometimes-not-select-the-first-completion-on-the-list--why-is-there-an-empty-entry-at-the-beginning-of-the-completion-list--what-happened-to-old-style-default-selection.
                              (car headers)))
                     (info (cl-find header .HeaderInfos :key #'header :test #'string=)))
                (unless info (user-error "No header selected"))
                (setcar .HeaderInfos info)
                (setcdr .HeaderInfos nil))))
        (mapc #'delete-overlay overlays)))))

(defun clang-include-fixer--symbol-name (symbol-infos)
  "Return the unique symbol name in SYMBOL-INFOS.
Raise a signal if the symbol name is not unique."
  (let ((symbols (delete-dups (mapcar (lambda (info)
                                        (let-alist info .RawIdentifier))
                                      symbol-infos))))
    (when (cdr symbols)
      (error "Multiple symbols %s returned" symbols))
    (car symbols)))

(defun clang-include-fixer--highlight (symbol-info)
  "Add an overlay to highlight SYMBOL-INFO, if it points to a non-empty range.
Return the overlay object, or nil."
  (let-alist symbol-info
    (unless (zerop .Range.Length)
      (let ((overlay (make-overlay
                      (clang-include-fixer--filepos-to-bufferpos
                       .Range.Offset 'approximate)
                      (clang-include-fixer--filepos-to-bufferpos
                       (+ .Range.Offset .Range.Length) 'approximate))))
        (overlay-put overlay 'face 'clang-include-fixer-highlight)
        overlay))))

(defun clang-include-fixer--closest-overlay (overlays)
  "Return the start of the overlay in OVERLAYS that is closest to point."
  (cl-check-type overlays cons)
  (let ((point (point))
        acc)
    (dolist (overlay overlays acc)
      (let ((start (overlay-start overlay)))
        (when (or (null acc) (< (abs (- point start)) (abs (- point acc))))
          (setq acc start))))))

(defun clang-include-fixer--parse-json (buffer)
  "Parse a JSON response from clang-include-fixer in BUFFER.
Return the JSON object as an association list."
  (with-current-buffer buffer
    (save-excursion
      (goto-char (point-min))
      (let ((json-object-type 'alist)
            (json-array-type 'list)
            (json-key-type 'symbol)
            (json-false :json-false)
            (json-null nil)
            (json-pre-element-read-function nil)
            (json-post-element-read-function nil))
        (json-read)))))

(defun clang-include-fixer--encode-json (object)
  "Return the JSON representation of OBJECT as a string."
  (let ((json-encoding-separator ",")
        (json-encoding-default-indentation "  ")
        (json-encoding-pretty-print nil)
        (json-encoding-lisp-style-closings nil)
        (json-encoding-object-sort-predicate nil))
    (json-encode object)))

(defun clang-include-fixer--symbol-at-point ()
  "Return the qualified symbol at point.
If there is no symbol at point, return nil."
  ;; Let ‘bounds-of-thing-at-point’ to do the hard work and deal with edge
  ;; cases.
  (let ((bounds (bounds-of-thing-at-point 'symbol)))
    (when bounds
      (let ((beg (car bounds))
            (end (cdr bounds)))
        (save-excursion
          ;; Extend the symbol range to the left.  Skip over namespace
          ;; delimiters and parent namespace names.
          (goto-char beg)
          (while (and (clang-include-fixer--skip-double-colon-backward)
                      (skip-syntax-backward "w_")))
          ;; Skip over one more namespace delimiter, for absolute names.
          (clang-include-fixer--skip-double-colon-backward)
          (setq beg (point))
          ;; Extend the symbol range to the right.  Skip over namespace
          ;; delimiters and child namespace names.
          (goto-char end)
          (while (and (clang-include-fixer--skip-double-colon-forward)
                      (skip-syntax-forward "w_")))
          (setq end (point)))
        (buffer-substring-no-properties beg end)))))

(defun clang-include-fixer--skip-double-colon-forward ()
  "Skip a double colon.
When the next two characters are '::', skip them and return
non-nil.  Otherwise return nil."
  (let ((end (+ (point) 2)))
    (when (and (<= end (point-max))
               (string-equal (buffer-substring-no-properties (point) end) "::"))
      (goto-char end)
      t)))

(defun clang-include-fixer--skip-double-colon-backward ()
  "Skip a double colon.
When the previous two characters are '::', skip them and return
non-nil.  Otherwise return nil."
  (let ((beg (- (point) 2)))
    (when (and (>= beg (point-min))
               (string-equal (buffer-substring-no-properties beg (point)) "::"))
      (goto-char beg)
      t)))

;; ‘filepos-to-bufferpos’ is new in Emacs 25.1.  Provide a fallback for older
;; versions.
(defalias 'clang-include-fixer--filepos-to-bufferpos
  (if (fboundp 'filepos-to-bufferpos)
      'filepos-to-bufferpos
    (lambda (byte &optional _quality _coding-system)
      (byte-to-position (1+ byte)))))

;; ‘format-message’ is new in Emacs 25.1.  Provide a fallback for older
;; versions.
(defalias 'clang-include-fixer--format-message
  (if (fboundp 'format-message) 'format-message 'format))

;; ‘file-local-name’ is new in Emacs 26.1.  Provide a fallback for older
;; versions.
(defalias 'clang-include-fixer--file-local-name
  (if (fboundp 'file-local-name) #'file-local-name
    (lambda (file) (or (file-remote-p file 'localname) file))))

(provide 'clang-include-fixer)
;;; clang-include-fixer.el ends here
