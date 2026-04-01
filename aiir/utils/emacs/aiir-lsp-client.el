;;; aiir-lsp-clinet.el --- LSP clinet for the AIIR.

;; Copyright (C) 2022 The AIIR Authors.
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

;; Version: 0.1.0

;;; Commentary:

;; LSP clinet to use with `aiir-mode' that uses `aiir-lsp-server' or any
;; user made compatible server.

;;; Code:
(require 'lsp-mode)

(defgroup lsp-aiir nil
  "LSP support for AIIR."
  :group 'lsp-mode
  :link '(url-link "https://aiir.llvm.org/docs/Tools/AIIRLSP/"))


(defcustom lsp-aiir-server-executable "aiir-lsp-server"
  "Command to start the aiir language server."
  :group 'lsp-aiir
  :risky t
  :type 'file)


(defun lsp-aiir-setup ()
  "Setup the LSP client for AIIR."
  (add-to-list 'lsp-language-id-configuration '(aiir-mode . "aiir"))

  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection (lambda () lsp-aiir-server-executable))
    :activation-fn (lsp-activate-on "aiir")
    :priority -1
    :server-id 'aiir-lsp)))


(provide 'aiir-lsp-client)
;;; aiir-lsp-client.el ends here
