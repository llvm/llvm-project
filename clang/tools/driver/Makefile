##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ../..

TOOLNAME = clang
TOOLALIAS = clang++

ifdef CLANG_ORDER_FILE
TOOL_ORDER_FILE := $(CLANG_ORDER_FILE)
endif

# Include tool version information on OS X.
TOOL_INFO_PLIST := Info.plist

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

# Have the option of not supporting plugins. This is important for startup
# performance.
ifeq ($(CLANG_PLUGIN_SUPPORT), 1)
NO_DEAD_STRIP := 1
else
TOOL_NO_EXPORTS := 1
endif

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter codegen \
                   instrumentation ipo irreader linker objcarcopts option \
                   profiledata selectiondag
USEDLIBS = clangFrontendTool.a clangFrontend.a clangDriver.a \
           clangSerialization.a clangCodeGen.a clangParse.a clangSema.a \
           clangRewriteFrontend.a clangRewrite.a

ifeq ($(ENABLE_CLANG_STATIC_ANALYZER),1)
USEDLIBS += clangStaticAnalyzerFrontend.a clangStaticAnalyzerCheckers.a \
            clangStaticAnalyzerCore.a
endif

ifeq ($(ENABLE_CLANG_ARCMT),1)
USEDLIBS += clangARCMigrate.a
endif

USEDLIBS += clangAnalysis.a clangEdit.a clangAST.a clangLex.a \
            clangAPINotes.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

# Set the tool version information values.
ifeq ($(HOST_OS),Darwin)
ifdef CLANG_VENDOR
TOOL_INFO_NAME := $(CLANG_VENDOR) clang
else
TOOL_INFO_NAME := clang
endif

ifdef CLANG_VENDOR_UTI
TOOL_INFO_UTI := $(CLANG_VENDOR_UTI)
else
TOOL_INFO_UTI := org.llvm.clang
endif

TOOL_INFO_VERSION := $(word 3,$(shell grep "CLANG_VERSION " \
	$(PROJ_OBJ_DIR)/$(CLANG_LEVEL)/include/clang/Basic/Version.inc))
ifdef LLVM_SUBMIT_VERSION
TOOL_INFO_BUILD_VERSION := $(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION)
else
TOOL_INFO_BUILD_VERSION := 
endif
endif
