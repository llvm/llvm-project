# Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

# Makefile for installing libc message catalogs.

subdir := po

include ../Makeconfig

# Add names of the languages with broken .po files here.
BROKEN_LINGUAS =

# List of languages for which we have message catalogs of translations.
ALL_LINGUAS := $(filter-out $(BROKEN_LINGUAS),$(basename $(wildcard *.po)))

# You can override this in configparms or the make command line to limit
# the languages which get installed.
ifdef LINGUAS
LINGUAS := $(filter $(addsuffix %,$(LINGUAS)),$(ALL_LINGUAS))
else
LINGUAS = $(ALL_LINGUAS)
endif

# Text domain name to install under; must match ../locale/SYS_libc.c string.
domainname = libc

# Pattern for where message catalog object for language % gets installed.
mo-installed = $(inst_localedir)/%/LC_MESSAGES/$(domainname).mo

# Files to install: a $(domainname).mo file for each language.
install-others = $(LINGUAS:%=$(mo-installed))

generated += $(ALL_LINGUAS:%=%.mo)


include ../Rules

# Generate the translation template from all the source files.
libc.pot: pot.header libc.pot.files
	@rm -f $@.new
	set `date -R`; disp="$$6"; \
	sed -e 's/VERSION/$(version)/' \
	    -e "s/DATE/`date +'%Y-%m-%d %H:%M'$$disp`/" \
	    -e "s/CHARSET/UTF-8/" \
	    $< > $@.new
	egrep -v '\.ksh$$' $(word 2,$^) > $(objdir)/tmp-libc.pot-files
	cd ..; $(XGETTEXT) --keyword=_ --keyword=N_ \
			   --add-comments=TRANS --flag=error:3:c-format \
			   --flag=f_print:2:c-format \
			   --sort-by-file --omit-header -E -n -d - \
			   -f $(objdir)/tmp-libc.pot-files >> po/$@.new
	egrep '\.ksh$$' $(word 2,$^) > $(objdir)/tmp-libc.pot-files
	cd ..; $(XGETTEXT) --add-comments=TRANS --language=Shell \
			   --sort-by-file --omit-header -E -n -o po/$@.new -j \
			   -f $(objdir)/tmp-libc.pot-files po/$@.new
	rm $(objdir)/tmp-libc.pot-files
	mv -f $@.new $@

po-sed-cmd = \
  '/\/tst-/d;$(foreach S,[ch] cxx sh ksh bash,$(subst .,\.,/.$S\(.in\)*$$/p;))'

# Omit tst-* files, and get only files with the given suffixes.
libc.pot.files: FORCE
	$(..)scripts/list-sources.sh .. | sed -n $(po-sed-cmd) > $@.new
	mv -f $@.new $@

.SUFFIXES: .mo .po .pot

# Compile the binary message object files from the portable object source
# files of translations for each language.
$(objpfx)%.mo: %.po
	$(make-target-directory)
	$(MSGFMT) -o $@ $<

# Install the message object files as libc.po in the language directory.
$(mo-installed): $(objpfx)%.mo $(+force); $(do-install) || exit 0

# Update translations from the translation project.
TRANSLATIONS_URL = https://translationproject.org/latest/libc
WGET = wget

update-translations:
	$(WGET) -qO- $(TRANSLATIONS_URL) | sed -n 's|.*href="\([^"]\+po\)".*|\1|p' | \
	  while read f; do \
	    $(WGET) -O $(objdir)/$$f-tmp2 $(TRANSLATIONS_URL)/$$f && \
	    msgmerge --previous --no-wrap $(objdir)/$$f-tmp2 libc.pot > \
	      $(objdir)/$$f-tmp && \
	    msgattrib --translated --no-fuzzy --no-obsolete --no-wrap \
	      $(objdir)/$$f-tmp > $(objdir)/$$f-tmp0 && \
	    mv -f $(objdir)/$$f-tmp0 $$f; \
	  done
