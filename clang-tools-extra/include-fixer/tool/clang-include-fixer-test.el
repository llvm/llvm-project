;;; clang-include-fixer-test.el --- unit tests for clang-include-fixer.el  -*- lexical-binding: t; -*-

;;; Commentary:

;; Unit tests for clang-include-fixer.el.

;;; Code:

(require 'cc-mode)
(require 'ert)

(ert-deftest clang-include-fixer--insert-line ()
  "Unit test for `clang-include-fixer--insert-line'."
  (with-temp-buffer
    (insert "aa\nab\nac\nad\n")
    (let ((from (current-buffer)))
      (with-temp-buffer
        (insert "aa\nac\nad\n")
        (let ((to (current-buffer)))
          (should (clang-include-fixer--insert-line from to))
          (should (equal (buffer-string) "aa\nab\nac\nad\n")))))
    (should (equal (buffer-string) "aa\nab\nac\nad\n"))))

(ert-deftest clang-include-fixer--symbol-at-point ()
  "Unit test for `clang-include-fixer--symbol-at-point'."
  (with-temp-buffer
    (insert "a+bbb::cc")
    (c++-mode)
    (goto-char (point-min))
    (should (equal (clang-include-fixer--symbol-at-point) "a"))
    (forward-char)
    ;; Emacs treats the character immediately following a symbol as part of the
    ;; symbol.
    (should (equal (clang-include-fixer--symbol-at-point) "a"))
    (forward-char)
    (should (equal (clang-include-fixer--symbol-at-point) "bbb::cc"))
    (goto-char (point-max))
    (should (equal (clang-include-fixer--symbol-at-point) "bbb::cc"))))

;;; clang-include-fixer-test.el ends here
