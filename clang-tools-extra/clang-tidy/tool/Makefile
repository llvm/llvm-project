##===- clang-tidy/tool/Makefile ----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../../..

TOOLNAME = clang-tidy

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc option
USEDLIBS = clangTidy.a clangTidyLLVMModule.a clangTidyGoogleModule.a \
	   clangTidyMiscModule.a clangTidyReadability.a clangTidyUtils.a \
	   clangStaticAnalyzerFrontend.a clangStaticAnalyzerCheckers.a \
	   clangStaticAnalyzerCore.a \
	   clangFormat.a clangASTMatchers.a clangTooling.a clangToolingCore.a \
	   clangFrontend.a clangSerialization.a clangDriver.a clangParse.a \
	   clangSema.a clangAnalysis.a clangRewriteFrontend.a clangRewrite.a \
	   clangEdit.a clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

PROJ_sharedir := $(DESTDIR)$(PROJ_prefix)/share/clang

FILESLIST := $(notdir $(wildcard $(PROJ_SRC_DIR)/*.py))

SRCFILES := $(addprefix $(PROJ_SRC_DIR)/, $(FILESLIST))
DESTFILES := $(addprefix $(PROJ_sharedir)/, $(FILESLIST))

$(PROJ_sharedir):
	$(Echo) Making install directory: $@
	$(Verb) $(MKDIR) $@

$(DESTFILES): $(SRCFILES) $(PROJ_sharedir)

$(PROJ_sharedir)/%.py: $(PROJ_SRC_DIR)/%.py
	$(Echo) Installing script file: $(notdir $<)
	$(Verb) $(ScriptInstall) $< $(PROJ_sharedir)

install-local:: $(DESTFILES)
