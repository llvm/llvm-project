;;; aiir-mode.el --- Major mode for the AIIR assembler language -*- lexical-binding: t -*-

;; Copyright (C) 2019 The AIIR Authors.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;;      http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

;; Version: 0.2.0
;; Homepage: https://aiir.llvm.org/
;; Package-Requires: ((emacs "24.3"))

;;; Commentary:

;; Major mode for editing AIIR files.

;;; Code:

(defvar aiir-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?% "_" table)
    (modify-syntax-entry ?@ "_" table)
    (modify-syntax-entry ?# "_" table)
    (modify-syntax-entry ?. "_" table)
    (modify-syntax-entry ?/ ". 12" table)
    (modify-syntax-entry ?\n "> " table)
    table)
  "Syntax table used while in AIIR mode.")

(defvar aiir-font-lock-keywords
  (list
   ;; Variables
   '("%[-a-zA-Z$._0-9]*" . font-lock-variable-name-face)
   ;; Functions
   '("@[-a-zA-Z$._0-9]*" . font-lock-function-name-face)
   ;; Affinemaps
   '("#[-a-zA-Z$._0-9]*" . font-lock-variable-name-face)
   ;; Types
   '("\\b\\(f16\\|bf16\\|f32\\|f64\\|index\\|tf_control\\|i[1-9][0-9]*\\)\\b" . font-lock-type-face)
   '("\\b\\(tensor\\|vector\\|memref\\)\\b" . font-lock-type-face)
   ;; Dimension lists
   '("\\b\\([0-9?]+x\\)*\\(f16\\|bf16\\|f32\\|f64\\|index\\|i[1-9][0-9]*\\)\\b" . font-lock-preprocessor-face)
   ;; Integer literals
   '("\\b[-]?[0-9]+\\b" . font-lock-preprocessor-face)
   ;; Floating point constants
   '("\\b[-+]?[0-9]+.[0-9]*\\([eE][-+]?[0-9]+\\)?\\b" . font-lock-preprocessor-face)
   ;; Hex constants
   '("\\b0x[0-9A-Fa-f]+\\b" . font-lock-preprocessor-face)
   ;; Keywords
   `(,(regexp-opt
       '(;; Toplevel entities
         "br" "ceildiv" "func" "cond_br" "else" "extfunc" "false" "floordiv" "for" "if" "mod" "return" "size" "step" "to" "true" "??" ) 'symbols) . font-lock-keyword-face))
  "Syntax highlighting for AIIR.")

;;;###autoload
(define-derived-mode aiir-mode prog-mode "AIIR"
  "Major mode for editing AIIR source files.
\\{aiir-mode-map}
  Runs `aiir-mode-hook' on startup."
  (setq font-lock-defaults `(aiir-font-lock-keywords))
  (setq-local comment-start "//"))

;; Associate .aiir files with aiir-mode
;;;###autoload
(add-to-list 'auto-mode-alist (cons "\\.aiir\\'" 'aiir-mode))

(defgroup aiir nil
  "Major mode for editing AIIR source files."
  :group 'languages
  :prefix "aiir-")

;; Set default value of opt-tool to use as aiir-opt.
(defcustom aiir-opt "aiir-opt"
  "Commandline AIIR opt tool to use."
  :type 'string)

;; Enable reading/writing .aiirbc files.
(require 'jka-compr)
(add-to-list 'jka-compr-compression-info-list
  (vector "\\.aiirbc\\'"
   "aiir-to-bytecode" aiir-opt (vector "--aiir-print-debuginfo" "--emit-bytecode" "-o" "-" "-")
   "aiir-bytecode-to-text" aiir-opt (vector "--aiir-print-debuginfo")
   nil nil "ML\357R"))
(jka-compr-update)
(auto-compression-mode t)
(add-to-list 'auto-mode-alist (cons "\\.aiirbc\\'" 'aiir-mode))

(provide 'aiir-mode)
;;; aiir-mode.el ends here
