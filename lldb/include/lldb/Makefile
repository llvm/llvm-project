LEVEL = ../../../..
DIRS :=

include $(LEVEL)/Makefile.common

install-local::
	$(Echo) Installing Clang include files
	$(Verb) $(MKDIR) $(DESTDIR)$(PROJ_includedir)
	$(Verb) if test -d "$(PROJ_SRC_ROOT)/tools/clang/include/clang" ; then \
	  cd $(PROJ_SRC_ROOT)/tools/clang/include && \
	  for  hdr in `find clang -type f '!' '(' -name '*~' \
	      -o -name '.#*' -o -name '*.in' -o -name '*.txt' \
	      -o -name 'Makefile' -o -name '*.td' -o -name '*.orig' ')' -print \
              | grep -v CVS | grep -v .svn | grep -v .dir` ; do \
	    instdir=$(DESTDIR)`dirname "$(PROJ_includedir)/$$hdr"` ; \
	    if test \! -d "$$instdir" ; then \
	      $(EchoCmd) Making install directory $$instdir ; \
	      $(MKDIR) $$instdir ;\
	    fi ; \
	    $(DataInstall) $$hdr $(DESTDIR)$(PROJ_includedir)/$$hdr ; \
	  done ; \
	fi
ifneq ($(PROJ_SRC_ROOT),$(PROJ_OBJ_ROOT))
	$(Verb) if test -d "$(PROJ_OBJ_ROOT)/tools/clang/include/clang" ; then \
	  cd $(PROJ_OBJ_ROOT)/tools/clang/include && \
	  for hdr in `find clang -type f '!' '(' -name 'Makefile' ')' -print \
            | grep -v CVS | grep -v .tmp | grep -v .dir` ; do \
	    $(DataInstall) $$hdr $(DESTDIR)$(PROJ_includedir)/$$hdr ; \
	  done ; \
	fi
endif
