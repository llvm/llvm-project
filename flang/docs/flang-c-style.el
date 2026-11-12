;;===-- docs/flang-c-style.el ------------------------------------===;;
;;
;; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
;; See https://llvm.org/LICENSE.txt for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;;
;;===----------------------------------------------------------------------===;;

;; Define a cc-mode style for editing C++ codes in Flang.
;;
;; Inspired from LLVM style in
;;    https://github.com/llvm/llvm-project/blob/main/llvm/utils/emacs/emacs.el
;;

(c-add-style "flang"
             '("gnu"
	       (fill-column . 80)
	       (c++-indent-level . 2)
	       (c-basic-offset . 2)
	       (indent-tabs-mode . nil)
	       (c-offsets-alist .
                   ((arglist-intro . ++) 
                    (innamespace . 0)
                    (member-init-intro . ++)
                    ))
               ))


;;
;; Use the following to make it the default.  
;;

(defun flang-c-mode-hook ()
  (c-set-style "flang")
  )

(add-hook 'c-mode-hook   'flang-c-mode-hook)
(add-hook 'c++-mode-hook 'flang-c-mode-hook)
