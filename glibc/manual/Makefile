# Copyright (C) 1992-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

# Makefile for the GNU C Library manual.

subdir := manual

include ../Makeconfig

.PHONY: dvi pdf info html

dvi: $(objpfx)libc.dvi
pdf: $(objpfx)libc.pdf

TEXI2DVI = texi2dvi
TEXI2PDF = texi2dvi --pdf

ifneq ($(strip $(MAKEINFO)),:)
info: $(objpfx)libc.info
endif

chapters = $(addsuffix .texi, \
		       intro errno memory ctype string charset locale	\
		       message search pattern io stdio llio filesys	\
		       pipe socket terminal syslog math arith time	\
		       resource setjmp signal startup process ipc job	\
		       nss users sysinfo conf crypt debug threads	\
		       probes tunables)
appendices = lang.texi header.texi install.texi maint.texi platform.texi \
	     contrib.texi
licenses = freemanuals.texi lgpl-2.1.texi fdl-1.3.texi

-include $(objpfx)texis
$(objpfx)texis: texis.awk $(chapters) $(appendices) $(licenses)
	$(make-target-directory)
	$(AWK) -f $^ > $@.T
	mv -f $@.T $@

nonexamples = $(filter-out %.c.texi, $(texis))
examples = $(filter %.c.texi, $(texis))

# Generated files directly included from libc.texinfo.
libc-texi-generated = chapters.texi top-menu.texi dir-add.texi \
		      libm-err.texi version.texi pkgvers.texi

# Add path to build dir for generated files
texis-path := $(filter-out $(libc-texi-generated) summary.texi $(examples), \
				$(texis)) 				    \
	 $(addprefix $(objpfx),$(filter $(libc-texi-generated) summary.texi \
					$(examples), $(texis)))

# Kludge: implicit rule so Make knows the one command does it all.
chapters.% top-menu.%: libc-texinfo.sh $(texis-path) Makefile
	AWK=$(AWK) $(SHELL) $< $(objpfx) \
				'$(chapters)' \
			       '$(appendices) $(licenses)'


$(objpfx)libc.dvi $(objpfx)libc.pdf $(objpfx)libc.info: \
	$(addprefix $(objpfx),$(libc-texi-generated))
$(objpfx)libc.dvi $(objpfx)libc.pdf: texinfo.tex

html: $(objpfx)libc/index.html
$(objpfx)libc/index.html: $(addprefix $(objpfx),$(libc-texi-generated))
	$(MAKEINFO) -P $(objpfx) -o $(objpfx)libc --html libc.texinfo

# Generate the summary from the Texinfo source files for each chapter.
$(objpfx)summary.texi: $(objpfx)stamp-summary ;
$(objpfx)stamp-summary: summary.pl $(filter-out $(objpfx)summary.texi, \
					$(texis-path))
	$(SHELL) ./check-safety.sh $(filter-out $(objpfx)%, $(texis-path))
	LC_ALL=C $(PERL) $^ > $(objpfx)summary-tmp
	$(move-if-change) $(objpfx)summary-tmp $(objpfx)summary.texi
	touch $@

# Generate a file which can be added to the `dir' content to provide direct
# access to the documentation of the function, variables, and other
# definitions.
$(objpfx)dir-add.texi: xtract-typefun.awk $(texis-path)
	(echo "@dircategory GNU C library functions and macros";	\
	 echo "@direntry";						\
	 $(AWK) -f $^ | sort;						\
	 echo "@end direntry") > $@.new
	mv -f $@.new $@

# The table with the math errors is generated.
$(objpfx)libm-err.texi: $(objpfx)stamp-libm-err
$(objpfx)stamp-libm-err: $(..)math/gen-libm-test.py \
			 $(wildcard $(foreach dir,$(sysdirs),\
						  $(dir)/libm-test-ulps))
	pwd=`pwd`; \
	$(PYTHON) $< -s $$pwd/.. -m $(objpfx)libm-err-tmp
	$(move-if-change) $(objpfx)libm-err-tmp $(objpfx)libm-err.texi
	touch $@

# Package version and bug reporting URL.
$(objpfx)pkgvers.texi: $(objpfx)stamp-pkgvers ;
$(objpfx)stamp-pkgvers: $(common-objpfx)config.make
	echo "@ifclear PKGVERS" > $(objpfx)pkgvers-tmp
	echo "@set PKGVERS" >> $(objpfx)pkgvers-tmp
	echo "@set PKGVERSION $(PKGVERSION_TEXI)" >> $(objpfx)pkgvers-tmp
	if [ "$(PKGVERSION_TEXI)" = "(GNU libc) " ]; then \
	  echo "@set PKGVERSION_DEFAULT" >> $(objpfx)pkgvers-tmp; \
	fi
	echo "@set REPORT_BUGS_TO $(REPORT_BUGS_TEXI)" >> $(objpfx)pkgvers-tmp
	echo "@end ifclear" >> $(objpfx)pkgvers-tmp
	$(move-if-change) $(objpfx)pkgvers-tmp $(objpfx)pkgvers.texi
	touch $@

# Generate a file with the version number.
$(objpfx)version.texi: $(objpfx)stamp-version ;
$(objpfx)stamp-version: $(common-objpfx)config.make
	echo "@set VERSION $(version)" > $(objpfx)version-tmp
	$(move-if-change) $(objpfx)version-tmp $(objpfx)version.texi
	touch $@

# Generate Texinfo files from the C source for the example programs.
$(objpfx)%.c.texi: examples/%.c
	sed -e '1,/^\*\/$$/d'				\
	    -e 's,[{}],@&,g'				\
	    -e 's,/\*\(@.*\)\*/,\1,g'			\
	    -e 's,/\*  *,/* @r{,g' -e 's,  *\*/,} */,'	\
	    -e 's/\(@[a-z][a-z]*\)@{\([^}]*\)@}/\1{\2}/g'\
	    $< | expand > $@.new
	mv -f $@.new $@

$(objpfx)%.info: %.texinfo
	LANGUAGE=C LC_ALL=C $(MAKEINFO) -P $(objpfx) --output=$@ $<

$(objpfx)%.dvi: %.texinfo
	cd $(objpfx);$(TEXI2DVI) -I $(shell cd $(<D) && pwd) --output=$@ \
				$(shell cd $(<D) && pwd)/$<

$(objpfx)%.pdf: %.texinfo
	cd $(objpfx);$(TEXI2PDF) -I $(shell cd $(<D) && pwd) --output=$@ \
				$(shell cd $(<D) && pwd)/$<


# Distribution.
minimal-dist = summary.pl texis.awk tsort.awk libc-texinfo.sh libc.texinfo \
	       libm-err.texi stamp-libm-err check-safety.sh		    \
	       $(filter-out summary.texi, $(nonexamples))		    \
	       $(patsubst %.c.texi,examples/%.c, $(examples))

indices = cp fn pg tp vr ky
generated-dirs += libc
generated += libc.dvi libc.pdf libc.tmp libc.info*			      \
	     stubs							      \
	     texis summary.texi stamp-summary *.c.texi			      \
	     $(foreach index,$(indices),libc.$(index) libc.$(index)s)	      \
	     libc.log libc.aux libc.toc					      \
	     $(libc-texi-generated)					      \
	     stamp-libm-err stamp-version

include ../Rules

.PHONY: install subdir_install install-data
install-data subdir_install: install
# Generated files requiring python: libm-err.texi
# Generated files requiring perl: summary.texi
ifneq ($(PERL),no)
ifneq ($(strip $(MAKEINFO)),:)
install: $(inst_infodir)/libc.info
	@if $(SHELL) -c '$(INSTALL_INFO) --version' >/dev/null 2>&1; then \
	 test -f $(inst_infodir)/dir || $(INSTALL_DATA) dir $(inst_infodir);\
	 $(INSTALL_INFO) --info-dir=$(inst_infodir) $(inst_infodir)/libc.info;\
	else : ; fi
endif
endif
# Catchall implicit rule for other installation targets from the parent.
install-%: ;

$(inst_infodir)/libc.info: $(objpfx)libc.info
	$(make-target-directory)
	for file in $<*; do \
	  $(INSTALL_DATA) $$file $(@D)/; \
	done

TAGS: $(minimal-dist)
	$(ETAGS) -o $@ $^
