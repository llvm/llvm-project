##===- docs/tools/Makefile ---------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

ifdef BUILD_FOR_WEBSITE

# FIXME: This was copied from the CommandGuide makefile. Figure out
# how to get this stuff on the website.

# This special case is for keeping the CommandGuide on the LLVM web site
# up to date automatically as the documents are checked in. It must build
# the POD files to HTML only and keep them in the src directories. It must also
# build in an unconfigured tree, hence the ifdef. To use this, run
# make -s BUILD_FOR_WEBSITE=1 inside the cvs commit script.
SRC_DOC_DIR=
DST_HTML_DIR=html/
DST_MAN_DIR=man/man1/
DST_PS_DIR=ps/
CLANG_VERSION := trunk

# If we are in BUILD_FOR_WEBSITE mode, default to the all target.
all:: html man ps

clean:
	rm -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

# To create other directories, as needed, and timestamp their creation
%/.dir:
	-mkdir $* > /dev/null
	date > $@

else

# Otherwise, if not in BUILD_FOR_WEBSITE mode, use the project info.
CLANG_LEVEL := ../..
include $(CLANG_LEVEL)/Makefile

CLANG_VERSION := $(word 3,$(shell grep "CLANG_VERSION " \
	$(PROJ_OBJ_DIR)/$(CLANG_LEVEL)/include/clang/Basic/Version.inc))

SRC_DOC_DIR=$(PROJ_SRC_DIR)/
DST_HTML_DIR=$(PROJ_OBJ_DIR)/
DST_MAN_DIR=$(PROJ_OBJ_DIR)/
DST_PS_DIR=$(PROJ_OBJ_DIR)/

endif


POD  := $(wildcard $(SRC_DOC_DIR)*.pod)
HTML := $(patsubst $(SRC_DOC_DIR)%.pod, $(DST_HTML_DIR)%.html, $(POD))
MAN  := $(patsubst $(SRC_DOC_DIR)%.pod, $(DST_MAN_DIR)%.1, $(POD))
PS   := $(patsubst $(SRC_DOC_DIR)%.pod, $(DST_PS_DIR)%.ps, $(POD))

ifdef ONLY_MAN_DOCS
INSTALL_TARGETS := install-man
else
INSTALL_TARGETS := install-html install-man install-ps
endif

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

$(DST_HTML_DIR)%.html: %.pod $(DST_HTML_DIR)/.dir
	pod2html --css=manpage.css --htmlroot=. \
	  --podpath=. --infile=$< --outfile=$@ --title=$*

$(DST_MAN_DIR)%.1: %.pod $(DST_MAN_DIR)/.dir
	pod2man --release "clang $(CLANG_VERSION)" --center="Clang Tools Documentation" $< $@

$(DST_PS_DIR)%.ps: $(DST_MAN_DIR)%.1 $(DST_PS_DIR)/.dir
	groff -Tps -man $< > $@


html: $(HTML)
man: $(MAN)
ps: $(PS)

EXTRA_DIST := $(POD)

clean-local::
	$(Verb) $(RM) -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

HTML_DIR := $(DESTDIR)$(PROJ_docsdir)/html/clang
MAN_DIR  := $(DESTDIR)$(PROJ_mandir)/man1
PS_DIR   := $(DESTDIR)$(PROJ_docsdir)/ps

install-html:: $(HTML)
	$(Echo) Installing HTML Clang Tools Documentation
	$(Verb) $(MKDIR) $(HTML_DIR)
	$(Verb) $(DataInstall) $(HTML) $(HTML_DIR)
	$(Verb) $(DataInstall) $(PROJ_SRC_DIR)/manpage.css $(HTML_DIR)

install-man:: $(MAN)
	$(Echo) Installing MAN Clang Tools Documentation
	$(Verb) $(MKDIR) $(MAN_DIR)
	$(Verb) $(DataInstall) $(MAN) $(MAN_DIR)

install-ps:: $(PS)
	$(Echo) Installing PS Clang Tools Documentation
	$(Verb) $(MKDIR) $(PS_DIR)
	$(Verb) $(DataInstall) $(PS) $(PS_DIR)

install-local:: $(INSTALL_TARGETS)

uninstall-local::
	$(Echo) Uninstalling Clang Tools Documentation
	$(Verb) $(RM) -rf $(HTML_DIR) $(MAN_DIR) $(PS_DIR)

printvars::
	$(Echo) "POD            : " '$(POD)'
	$(Echo) "HTML           : " '$(HTML)'
