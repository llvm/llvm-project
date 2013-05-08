;;; Clang-format emacs integration for use with C/Objective-C/C++.

;; This defines a function clang-format-region that you can bind to a key.
;; A minimal .emacs would contain:
;;
;;   (load "<path-to-clang>/tools/clang-format/clang-format.el")
;;   (global-set-key [C-M-tab] 'clang-format-region)
;;
;; Depending on your configuration and coding style, you might need to modify
;; 'style' and 'binary' below.
(defun clang-format-region ()
  (interactive)

  (let* ((orig-windows (get-buffer-window-list (current-buffer)))
         (orig-window-starts (mapcar #'window-start orig-windows))
         (orig-point (point))
         (binary "clang-format")
         (style "LLVM"))
    (if mark-active
        (setq beg (region-beginning)
              end (region-end))
      (setq beg (min (line-beginning-position) (1- (point-max)))
            end (min (line-end-position) (1- (point-max)))))
    (call-process-region (point-min) (point-max) binary t t nil
                         "-offset" (number-to-string (1- beg))
                         "-length" (number-to-string (- end beg))
                         "-style" style)
    (goto-char orig-point)
    (dotimes (index (length orig-windows))
      (set-window-start (nth index orig-windows)
                        (nth index orig-window-starts)))))
