;;; clang-rename.el --- Renames every occurrence of a symbol found at <offset>.

;; Keywords: tools, c

;;; Commentary:

;; To install clang-rename.el make sure the directory of this file is in your
;; 'load-path' and add
;;
;;   (require 'clang-rename)
;;
;; to your .emacs configuration.

;;; Code:

(defcustom clang-rename-binary "clang-rename"
  "Path to clang-rename executable."
  :type 'hook
  :options '(turn-on-auto-fill flyspell-mode)
  :group 'wp)

(defun clang-rename (new-name)
  "Rename all instances of the symbol at the point using clang-rename"
  (interactive "sEnter a new name: ")
  (let (;; Emacs offset is 1-based.
        (offset (- (point) 1))
        (orig-buf (current-buffer))
        (file-name (buffer-file-name)))

    (let ((rename-command
          (format "bash -f -c '%s -offset=%s -new-name=%s -i %s'"
                               clang-rename-binary offset new-name file-name)))
          (message (format "Running clang-rename command %s" rename-command))
          ;; Run clang-rename via bash.
          (shell-command rename-command)
          ;; Reload buffer.
          (revert-buffer t t)
    )
  )
)

(provide 'clang-rename)

;;; clang-rename.el ends here
