##===- docs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ..
DIRS       := tools

ifdef BUILD_FOR_WEBSITE
PROJ_OBJ_DIR = .
DOXYGEN = doxygen

$(PROJ_OBJ_DIR)/doxygen.cfg: doxygen.cfg.in
	cat $< | sed \
	  -e 's/@abs_srcdir@/./g' \
	  -e 's/@DOT@/dot/g' \
	  -e 's/@PACKAGE_VERSION@/mainline/' \
	  -e 's/@abs_builddir@/./g' > $@
endif

include $(CLANG_LEVEL)/Makefile

HTML       := $(wildcard $(PROJ_SRC_DIR)/*.html) \
              $(wildcard $(PROJ_SRC_DIR)/*.css)
#IMAGES     := $(wildcard $(PROJ_SRC_DIR)/img/*.*)
DOXYFILES  := doxygen.cfg.in doxygen.css doxygen.footer doxygen.header \
              doxygen.intro
EXTRA_DIST := $(HTML) $(DOXYFILES) llvm.css CommandGuide img

.PHONY: install-html install-doxygen doxygen generated

install_targets :=
ifndef ONLY_MAN_DOCS
install_targets += install-html
endif
ifeq ($(ENABLE_DOXYGEN),1)
install_targets += install-doxygen
endif
install-local:: $(install_targets)

# Live documentation is generated for the web site using this target:
# 'make generated BUILD_FOR_WEBSITE=1'
generated:: doxygen

install-html: $(PROJ_OBJ_DIR)/html.tar.gz
	$(Echo) Installing HTML documentation
	$(Verb) $(MKDIR) $(DESTDIR)$(PROJ_docsdir)/html
	$(Verb) $(MKDIR) $(DESTDIR)$(PROJ_docsdir)/html/img
	$(Verb) $(DataInstall) $(HTML) $(DESTDIR)$(PROJ_docsdir)/html
#	$(Verb) $(DataInstall) $(IMAGES) $(DESTDIR)$(PROJ_docsdir)/html/img
	$(Verb) $(DataInstall) $(PROJ_OBJ_DIR)/html.tar.gz $(DESTDIR)$(PROJ_docsdir)

$(PROJ_OBJ_DIR)/html.tar.gz: $(HTML)
	$(Echo) Packaging HTML documentation
	$(Verb) $(RM) -rf $@ $(PROJ_OBJ_DIR)/html.tar
	$(Verb) cd $(PROJ_SRC_DIR) && \
	  $(TAR) cf $(PROJ_OBJ_DIR)/html.tar *.html
	$(Verb) $(GZIPBIN) $(PROJ_OBJ_DIR)/html.tar

install-doxygen: doxygen
	$(Echo) Installing doxygen documentation
	$(Verb) $(MKDIR) $(DESTDIR)$(PROJ_docsdir)/html/doxygen
	$(Verb) $(DataInstall) $(PROJ_OBJ_DIR)/doxygen.tar.gz $(DESTDIR)$(PROJ_docsdir)
	$(Verb) cd $(PROJ_OBJ_DIR)/doxygen && \
	  $(FIND) . -type f -exec \
	    $(DataInstall) {} $(DESTDIR)$(PROJ_docsdir)/html/doxygen \;

doxygen: regendoc $(PROJ_OBJ_DIR)/doxygen.tar.gz

regendoc:
	$(Echo) Building doxygen documentation
	$(Verb) if test -e $(PROJ_OBJ_DIR)/doxygen ; then \
	  $(RM) -rf $(PROJ_OBJ_DIR)/doxygen ; \
	fi
	$(Verb) $(DOXYGEN) $(PROJ_OBJ_DIR)/doxygen.cfg

$(PROJ_OBJ_DIR)/doxygen.tar.gz: $(DOXYFILES) $(PROJ_OBJ_DIR)/doxygen.cfg
	$(Echo) Packaging doxygen documentation
	$(Verb) $(RM) -rf $@ $(PROJ_OBJ_DIR)/doxygen.tar
	$(Verb) $(TAR) cf $(PROJ_OBJ_DIR)/doxygen.tar doxygen
	$(Verb) $(GZIPBIN) $(PROJ_OBJ_DIR)/doxygen.tar
	$(Verb) $(CP) $(PROJ_OBJ_DIR)/doxygen.tar.gz $(PROJ_OBJ_DIR)/doxygen/html/

userloc: $(LLVM_SRC_ROOT)/docs/userloc.html

$(LLVM_SRC_ROOT)/docs/userloc.html:
	$(Echo) Making User LOC Table
	$(Verb) cd $(LLVM_SRC_ROOT) ; ./utils/userloc.pl -details -recurse \
	  -html lib include tools runtime utils examples autoconf test > docs/userloc.html

uninstall-local::
	$(Echo) Uninstalling Documentation
	$(Verb) $(RM) -rf $(DESTDIR)$(PROJ_docsdir)
