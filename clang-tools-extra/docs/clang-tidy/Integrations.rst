==================================
Clang-tidy IDE/Editor Integrations
==================================

.. _Clangd: https://clang.llvm.org/extra/clangd.html

Apart from being a standalone tool, :program:`clang-tidy` is integrated into
various IDEs, code analyzers, and editors. Besides, it is currently being
integrated into Clangd_. The following table shows the most
well-known :program:`clang-tidy` integrations in detail.

+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|                                      |        Feature                                                                                                                                           |
+======================================+========================+=================================+==========================+=========================================+==========================+
|  **Tool**                            | On-the-fly inspection  | Check list configuration (GUI)  | Options to checks (GUI)  | Configuration via ``.clang-tidy`` files | Custom clang-tidy binary |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|A.L.E. for Vim                        |         \+\            |               \-\               |           \-\            |                 \-\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|Clang Power Tools for Visual Studio   |         \-\            |               \+\               |           \-\            |                 \+\                     |           \-\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|Clangd                                |         \+\            |               \-\               |           \-\            |                 \-\                     |           \-\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|CLion IDE                             |         \+\            |               \+\               |           \+\            |                 \+\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|CodeChecker                           |         \-\            |               \-\               |           \-\            |                 \-\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|CPPCheck                              |         \-\            |               \-\               |           \-\            |                 \-\                     |           \-\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|CPPDepend                             |         \-\            |               \-\               |           \-\            |                 \-\                     |           \-\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|Flycheck for Emacs                    |         \+\            |               \-\               |           \-\            |                 \+\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|KDevelop IDE                          |         \-\            |               \+\               |           \+\            |                 \+\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|Qt Creator IDE                        |         \+\            |               \+\               |           \-\            |                 \-\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|ReSharper C++ for Visual Studio       |         \+\            |               \+\               |           \-\            |                 \+\                     |           \-\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|Syntastic for Vim                     |         \+\            |               \-\               |           \-\            |                 \-\                     |           \+\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+
|Visual Assist for Visual Studio       |         \+\            |               \+\               |           \-\            |                 \-\                     |           \-\            |
+--------------------------------------+------------------------+---------------------------------+--------------------------+-----------------------------------------+--------------------------+

**IDEs**

.. _CLion: https://www.jetbrains.com/clion/
.. _integrates clang-tidy: https://www.jetbrains.com/help/clion/clang-tidy-checks-support.html

CLion_ 2017.2 and later `integrates clang-tidy`_ as an extension to the
built-in code analyzer. Starting from 2018.2 EAP, CLion allows using
:program:`clang-tidy` via Clangd. Inspections and applicable quick-fixes are
performed on the fly, and checks can be configured in standard command line
format. In this integration, you can switch to the :program:`clang-tidy`
binary different from the bundled one, pass the configuration in
``.clang-tidy`` files instead of using the IDE settings, and configure
options for particular checks.

.. _KDevelop: https://www.kdevelop.org/
.. _kdev-clang-tidy: https://github.com/KDE/kdev-clang-tidy/

KDevelop_ with the kdev-clang-tidy_ plugin, starting from version 5.1, performs
static analysis using :program:`clang-tidy`. The plugin launches the
:program:`clang-tidy` binary from the specified location and parses its
output to provide a list of issues.

.. _QtCreator: https://www.qt.io/
.. _Clang Code Model: http://doc.qt.io/qtcreator/creator-clang-codemodel.html

QtCreator_ 4.6 integrates :program:`clang-tidy` warnings into the editor
diagnostics under the `Clang Code Model`_. To employ :program:`clang-tidy`
inspection in QtCreator, you need to create a copy of one of the presets and
choose the checks to be performed in the Clang Code Model Warnings menu.

.. _MS Visual Studio: https://visualstudio.microsoft.com/
.. _ReSharper C++: https://www.jetbrains.com/help/resharper/Clang_Tidy_Integration.html
.. _Visual Assist: https://docs.wholetomato.com/default.asp?W761
.. _Clang Power Tools: https://marketplace.visualstudio.com/items?itemName=caphyon.ClangPowerTools
.. _clang-tidy-vs: https://github.com/llvm-mirror/clang-tools-extra/tree/master/clang-tidy-vs

`MS Visual Studio`_ has a native clang-tidy-vs_ plugin and also can integrate
:program:`clang-tidy` by means of three other tools. The `ReSharper C++`_
extension, version 2017.3 and later, provides seamless :program:`clang-tidy`
integration: checks and quick-fixes run alongside native inspections. Apart
from that, ReSharper C++ incorporates :program:`clang-tidy` as a separate
step of its code clean-up process. `Visual Assist`_ build 2210 includes a
subset of :program:`clang-tidy` checklist to inspect the code as you edit.
Another way to bring :program:`clang-tidy` functionality to Visual Studio is
the `Clang Power Tools`_ plugin, which includes most of the
:program:`clang-tidy` checks and runs them during compilation or as a separate
step of code analysis.

**Editors**

.. _Flycheck: https://github.com/ch1bo/flycheck-clang-tidy
.. _Syntastic: https://github.com/vim-syntastic/syntastic
.. _A.L.E.: https://github.com/w0rp/ale
.. _Emacs24: https://www.gnu.org/s/emacs/
.. _Vim: https://www.vim.org/

Emacs24_, when expanded with the Flycheck_ plugin, incorporates the
:program:`clang-tidy` inspection into the syntax analyzer. For Vim_, you can
use Syntastic_, which includes :program:`clang-tidy`, or `A.L.E.`_,
a lint engine that applies :program:`clang-tidy` along with other linters.

**Analyzers**

.. _CPPDepend: https://www.cppdepend.com/cppdependv2018
.. _CPPCheck: https://sourceforge.net/p/cppcheck/news/
.. _CodeChecker: https://github.com/Ericsson/codechecker
.. _plugin: https://github.com/Ericsson/CodeCheckerEclipsePlugin

:program:`clang-tidy` is integrated in CPPDepend_ starting from version 2018.1
and CPPCheck_ 1.82. CPPCheck integration lets you import Visual Studio
solutions and run the :program:`clang-tidy` inspection on them. The
CodeChecker_ application of version 5.3 or later, which also comes as a plugin_
for Eclipse, supports :program:`clang-tidy` as a static analysis instrument and
allows to use a custom :program:`clang-tidy` binary.
