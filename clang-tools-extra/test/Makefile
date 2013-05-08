##===- tools/extra/test/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../..
include $(CLANG_LEVEL)/Makefile

# Test in all immediate subdirectories if unset.
ifdef TESTSUITE
TESTDIRS := $(TESTSUITE:%=$(PROJ_SRC_DIR)/%)
else
TESTDIRS ?= $(PROJ_SRC_DIR)
endif

# 'lit' wants objdir paths, so it will pick up the lit.site.cfg.
TESTDIRS := $(TESTDIRS:$(PROJ_SRC_DIR)%=$(PROJ_OBJ_DIR)%)

# Allow EXTRA_TESTDIRS to provide additional test directories.
TESTDIRS += $(EXTRA_TESTDIRS)

ifndef TESTARGS
ifdef VERBOSE
TESTARGS = -v
else
TESTARGS = -s -v
endif
endif

# Make sure any extra test suites can find the main site config.
LIT_ARGS := --param clang_site_config=$(PROJ_OBJ_DIR)/lit.site.cfg

ifdef VG
  LIT_ARGS += "--vg"
endif

all:: lit.site.cfg Unit/lit.site.cfg
	@ echo '--- Running the Clang extra tools tests for $(TARGET_TRIPLE) ---'
	@ $(PYTHON) $(LLVM_SRC_ROOT)/utils/lit/lit.py \
	  $(LIT_ARGS) $(TESTARGS) $(TESTDIRS)

FORCE:

lit.site.cfg: FORCE
	@echo "Making lit.site.cfg for Clang extra tools..."
	@$(ECHOPATH) s=@LLVM_SOURCE_DIR@=$(LLVM_SRC_ROOT)=g > lit.tmp
	@$(ECHOPATH) s=@LLVM_BINARY_DIR@=$(LLVM_OBJ_ROOT)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_TOOLS_DIR@=$(ToolDir)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_LIBS_DIR@=$(LibDir)=g >> lit.tmp
	@$(ECHOPATH) s=@CLANG_TOOLS_SOURCE_DIR@=$(PROJ_SRC_DIR)/..=g >> lit.tmp
	@$(ECHOPATH) s=@CLANG_TOOLS_BINARY_DIR@=$(PROJ_OBJ_DIR)/..=g >> lit.tmp
	@$(ECHOPATH) s=@TARGET_TRIPLE@=$(TARGET_TRIPLE)=g >> lit.tmp
	@sed -f lit.tmp $(PROJ_SRC_DIR)/lit.site.cfg.in > $@
	@-rm -f lit.tmp

Unit/lit.site.cfg: FORCE
	@echo "Making Unit/lit.site.cfg for Clang extra tools..."
	@$(MKDIR) $(dir $@)
	@$(ECHOPATH) s=@CLANG_TOOLS_BINARY_DIR@=$(PROJ_OBJ_DIR)/..=g >> lit.tmp
	@$(ECHOPATH) s=@TARGET_TRIPLE@=$(TARGET_TRIPLE)=g >> lit.tmp
	@$(ECHOPATH) s=@CLANG_TOOLS_SOURCE_DIR@=$(PROJ_SRC_DIR)/..=g >> lit.tmp
	@sed -f lit.tmp $(PROJ_SRC_DIR)/Unit/lit.site.cfg.in > $@
	@-rm -f lit.tmp

clean::
	@ find . -name Output | xargs rm -fr

.PHONY: all report clean
