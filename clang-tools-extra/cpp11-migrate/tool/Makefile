##===- tools/extra/loop-convert/Makefile ----sssss----------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TOOLNAME = cpp11-migrate

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

SOURCES = Cpp11Migrate.cpp

# FIXME: All these gross relative paths will go away once transforms are lib-ified.

# For each Transform subdirectory add to SOURCES and BUILT_SOURCES.
# BUILT_SOURCES ensures a subdirectory is created to house object files from
# transform subdirectories. See below for more on .objdir.
SOURCES += $(addprefix ../LoopConvert/,$(notdir $(wildcard $(PROJ_SRC_DIR)/../LoopConvert/*.cpp)))
BUILT_SOURCES = $(ObjDir)/../LoopConvert/.objdir
SOURCES += $(addprefix ../UseNullptr/,$(notdir $(wildcard $(PROJ_SRC_DIR)/../UseNullptr/*.cpp)))
BUILT_SOURCES += $(ObjDir)/../UseNullptr/.objdir
SOURCES += $(addprefix ../UseAuto/,$(notdir $(wildcard $(PROJ_SRC_DIR)/../UseAuto/*.cpp)))
BUILT_SOURCES += $(ObjDir)/../UseAuto/.objdir
SOURCES += $(addprefix ../AddOverride/,$(notdir $(wildcard $(PROJ_SRC_DIR)/../AddOverride/*.cpp)))
BUILT_SOURCES += $(ObjDir)/../AddOverride/.objdir

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc
USEDLIBS = clangTooling.a clangFrontend.a clangSerialization.a clangDriver.a \
		   clangRewriteFrontend.a clangRewriteCore.a clangParse.a \
		   clangSema.a clangAnalysis.a \
		   clangAST.a clangASTMatchers.a clangEdit.a clangLex.a clangBasic.a \
			 migrateCore.a

include $(CLANG_LEVEL)/Makefile

CPP.Flags += -I$(PROJ_SRC_DIR)/..

# BUILT_SOURCES gets used as a prereq for many top-level targets. However, at
# the point those targets are defined, $(ObjDir) hasn't been defined and so the
# directory to create becomes /<name>/ which is not what we want. So instead,
# this .objdir recipe is defined at at point where $(ObjDir) is defined and
# it's specialized to $(ObjDir) to ensure it only works on targets we want it
# to.
$(ObjDir)/%.objdir:
	$(Verb) $(MKDIR) $(ObjDir)/$* > /dev/null
	$(Verb) $(DOTDIR_TIMESTAMP_COMMAND) > $@

