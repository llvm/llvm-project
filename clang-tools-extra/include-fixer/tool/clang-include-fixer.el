;;; clang-include-fxier.el --- Emacs integration of the clang include fixer

;; Keywords: tools, c
;; Package-Requires: ((json "1.2"))

;;; Commentary:

;; This package allows to invoke the 'clang-include-fixer' within Emacs.
;; 'clang-include-fixer' provides an automated way of adding #include
;; directives for missing symbols in one translation unit, see
;; <http://clang.llvm.org/extra/include-fixer.html>.

;;; Code:

(require 'json)

(defgroup clang-include-fixer nil
  "Include fixer."
  :group 'tools)

(defcustom clang-include-fixer-executable
  "clang-include-fixer"
  "Location of the `clang-include-fixer' executable.

   A string containing the name or the full path of the executable."
  :group 'clang-include-fixer
  :type 'string
  :risky t)

(defcustom clang-include-fixer-input-format
  "yaml"
  "clang-include-fixer input format."
  :group 'clang-include-fixer
  :type 'string
  :risky t)

(defcustom clang-include-fixer-init-string
  ""
  "clang-include-fixer init string."
  :group 'clang-include-fixer
  :type 'string
  :risky t)

(defcustom clang-include-fixer-query-mode
  nil
  "clang-include-fixer query mode."
  :group 'clang-include-fixer
  :type 'boolean
  :risky t)

(defun clang-include-fixer-call-executable (callee
                                            include-fixer-parameter-a
                                            &optional include-fixer-parameter-b
                                            &optional include-fixer-parameter-c
                                            )
  "Calls clang-include-fixer with parameters INCLUDE-FIXER-PARAMETER-[ABC].
   If the call was successful the returned result is stored in a temp buffer
   and the function CALLEE is called on this temp buffer."

    (let ((temp-buffer (generate-new-buffer " *clang-include-fixer-temp*"))
          (temp-file (make-temp-file "clang-include-fixer")))
      (unwind-protect
          (let (status stderr operations)
            (if (eq include-fixer-parameter-c nil)
                (setq status
                      (call-process-region
                       (point-min) (point-max) clang-include-fixer-executable
                       nil `(,temp-buffer ,temp-file) nil

                       "-stdin"
                       include-fixer-parameter-a
                       (buffer-file-name)
                       ))
              (setq status
                    (call-process-region
                     (point-min) (point-max) clang-include-fixer-executable
                     nil `(,temp-buffer ,temp-file) nil

                     "-stdin"
                     include-fixer-parameter-a
                     include-fixer-parameter-b
                     include-fixer-parameter-c
                     (buffer-file-name)
                     )))

            (setq stderr
                  (with-temp-buffer
                    (insert-file-contents temp-file)
                    (when (> (point-max) (point-min))
                      (insert ": "))
                    (buffer-substring-no-properties
                     (point-min) (line-end-position))))

            (cond
             ((stringp status)
              (error "(clang-include-fixer killed by signal %s%s)" status
                     stderr))
             ((not (equal 0 status))
              (error "(clang-include-fixer failed with code %d%s)" status
                     stderr)))
            (funcall callee temp-buffer))
        (delete-file temp-file)
        (when (buffer-name temp-buffer) (kill-buffer temp-buffer)))))


(defun clang-include-fixer-replace_buffer (temp-buffer)
  "Replace current buffer by content of TEMP-BUFFER"

  (with-current-buffer temp-buffer
    (setq temp-start (point-min))
    (setq temp-end (point-max))
    )
  (barf-if-buffer-read-only)
  (erase-buffer)
  (save-excursion
    (insert-buffer-substring temp-buffer temp-start temp-end)))


(defun clang-include-fixer-add-header (temp-buffer)
  "Analyse the result of include-fixer stored in TEMP_BUFFER and add a
   missing header if there is any. If there are multiple possible headers
   the user can select one of them to be included."

  (with-current-buffer temp-buffer
    (setq result (buffer-substring (point-min) (point-max)))
    (setq include-fixer-context
          (let ((json-object-type 'plist))
            (json-read-from-string result))))

  ;; The header-infos is already sorted by include-fixer.
  (setq header-infos (plist-get include-fixer-context :HeaderInfos))
  (setq query-symbol-infos (plist-get include-fixer-context :QuerySymbolInfos))

  (if (eq 0 (length query-symbol-infos))
      (message "The file is fine, no need to add a header.")

    (setq symbol-info (elt query-symbol-infos 0))
    (setq symbol (plist-get symbol-info :RawIdentifier))
    (setq symbol-offset (plist-get (plist-get symbol-info :Range)
                                   :Offset))

    ;; Check the number of choices
    (if (eq 0 (length header-infos))
	(progn
	  (goto-char (1+ symbol-offset))
	  (message (concat "Couldn't find header for '" symbol "'.")))

      (setq symbol-length (plist-get (plist-get symbol-info :Range)
				     :Length))
      (goto-char (1+ symbol-offset))
      (setq symbol-overlay (make-overlay (1+ symbol-offset)
					 (+ symbol-offset symbol-length +1)))
      (overlay-put symbol-overlay 'face '(:background "green" :foreground
						      "black"))

      (message (number-to-string symbol-offset))
      (message (number-to-string symbol-length))

      (if (eq 1 (length header-infos))
	  (progn
	    (setq missing-header
		  (plist-get (elt header-infos 0) :Header))
	    (message (concat "Only one include is missing: "
			     missing-header )))

	;; Now iterate over vector and add items to list
	(setq include-list '())
	(setq index 0)
	(while (< index (length header-infos))
	  (setq entry (elt header-infos index))
	  (add-to-list 'include-list  (plist-get entry :Header))
	  (setq index (1+ index))
	  )

	(setq option-message (concat "Select include for '"
				     symbol
				     "' :"))
	(setq missing-header (ido-completing-read
                              option-message include-list)))

      ;; Now select set correct header info.
      (setq header-plist '())
      (setq index 0)
      (while (< index (length header-infos))
	(setq entry (elt header-infos index))
	(setq index (1+ index))
	(if (eq (plist-get entry :Header) missing-header)
	    (setq header-plist entry)))
      (setq include-fixer-context (plist-put
				   include-fixer-context
				   ':HeaderInfos (vector header-plist)))

      (clang-include-fixer-call-executable
       'clang-include-fixer-replace_buffer
       (concat "-insert-header=" (json-encode include-fixer-context)))
      (delete-overlay symbol-overlay))))


(defun clang-include-fixer ()
  "Invokes the Include Fixer to insert missing C++ headers."
  (interactive)

  (message (concat "Calling the include fixer. "
           "This might take some seconds. Please wait."))

  (if clang-include-fixer-query-mode
      (let (p1 p2)
      (save-excursion
        (skip-chars-backward "-a-zA-Z0-9_:")
        (setq p1 (point))
        (skip-chars-forward "-a-zA-Z0-9_:")
        (setq p2 (point))
        (setq query-symbol (buffer-substring-no-properties p1 p2))
        (if (string= "" query-symbol)
            (message "Skip querying empty symbol.")
          (clang-include-fixer-call-executable
            'clang-include-fixer-add-header
            (concat "-db=" clang-include-fixer-input-format)
            (concat "-input=" clang-include-fixer-init-string)
            (concat "-query-symbol=" (thing-at-point 'symbol))
          ))))
    (clang-include-fixer-call-executable
      'clang-include-fixer-add-header
      (concat "-db=" clang-include-fixer-input-format)
      (concat "-input=" clang-include-fixer-init-string)
      "-output-headers"))
  )

(provide 'clang-include-fixer)
;;; clang-include-fixer.el ends here
