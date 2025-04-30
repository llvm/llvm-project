# Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

# Makefile for generating and updating Unicode-extracted files.


# This Makefile is NOT used as part of the GNU libc build.  It needs
# to be run manually, within the source tree, at Unicode upgrades
# (change UNICODE_VERSION below), to update ../locales/i18n_ctype ctype
# information (part of the file is preserved, so don't wipe it all
# out), and ../charmaps/UTF-8.

# Use make all to generate the files used in the glibc build out of
# the original Unicode files; make check to verify that they are what
# we expect; make install to copy them to the location expected by the
# glibc build; and make clean to remove all generated files.

# We keep a local copy of the downloaded Unicode files, to avoid
# running afoul of the LGPL corresponding sources requirements, even
# though it's not clear that they are preferred over the generated
# files for making modifications.


UNICODE_VERSION = 13.0.0

PYTHON3 = python3
WGET = wget

DOWNLOADS = UnicodeData.txt DerivedCoreProperties.txt EastAsianWidth.txt PropList.txt
GENERATED = i18n_ctype tr_TR UTF-8 translit_combining translit_compat translit_circle translit_cjk_compat translit_font translit_fraction
REPORTS = i18n_ctype-report UTF-8-report

all: $(GENERATED)

check: check-i18n_ctype check-UTF-8

install:
	cp -p i18n_ctype ../locales/i18n_ctype
	cp -p tr_TR ../locales/tr_TR
	cp -p UTF-8 ../charmaps/UTF-8
	cp -p translit_combining ../locales/translit_combining
	cp -p translit_compat ../locales/translit_compat
	cp -p translit_circle ../locales/translit_circle
	cp -p translit_cjk_compat ../locales/translit_cjk_compat
	cp -p translit_font ../locales/translit_font
	cp -p translit_fraction ../locales/translit_fraction

clean: mostlyclean
	-rm -rf __pycache__
mostlyclean:
	-rm -f $(REPORTS) $(GENERATED)

.PHONY: all check clean mostlyclean install

i18n_ctype: UnicodeData.txt DerivedCoreProperties.txt
i18n_ctype: ../locales/i18n_ctype # Preserve non-ctype information.
i18n_ctype: gen_unicode_ctype.py
	$(PYTHON3) gen_unicode_ctype.py -u UnicodeData.txt \
	  -d DerivedCoreProperties.txt -i ../locales/i18n_ctype -o $@ \
	  --unicode_version $(UNICODE_VERSION)

i18n_ctype-report: i18n_ctype ../locales/i18n_ctype
i18n_ctype-report: ctype_compatibility.py ctype_compatibility_test_cases.py
	$(PYTHON3) ./ctype_compatibility.py -o ../locales/i18n_ctype \
	  -n i18n_ctype -a -m > $@

check-i18n_ctype: i18n_ctype-report
	@if grep '\(Missing\|Added\) [^0]\|^Number of errors[^=]* = [^0]' \
		i18n_ctype-report; \
	then echo manual verification required; false; else true; fi

tr_TR: UnicodeData.txt DerivedCoreProperties.txt
tr_TR: ../locales/tr_TR # Preserve non-ctype information.
tr_TR: gen_unicode_ctype.py
	$(PYTHON3) gen_unicode_ctype.py -u UnicodeData.txt \
	  -d DerivedCoreProperties.txt -i ../locales/tr_TR -o $@ \
	  --unicode_version $(UNICODE_VERSION) --turkish

UTF-8: UnicodeData.txt EastAsianWidth.txt
UTF-8: utf8_gen.py
	$(PYTHON3) utf8_gen.py -u UnicodeData.txt \
	-e EastAsianWidth.txt -p PropList.txt \
	--unicode_version $(UNICODE_VERSION)

UTF-8-report: UTF-8 ../charmaps/UTF-8
UTF-8-report: utf8_compatibility.py
	$(PYTHON3) ./utf8_compatibility.py -u UnicodeData.txt \
	-e EastAsianWidth.txt -o ../charmaps/UTF-8 \
	-n UTF-8 -a -m -c > $@

check-UTF-8: UTF-8-report
	@if grep '^Total.*: [^0]' UTF-8-report; \
	then echo manual verification required; false; else true; fi

translit_combining: UnicodeData.txt
translit_combining: gen_translit_combining.py
	$(PYTHON3) ./gen_translit_combining.py -u UnicodeData.txt \
	-o $@ --unicode_version $(UNICODE_VERSION)

translit_compat: UnicodeData.txt
translit_compat: gen_translit_compat.py
	$(PYTHON3) ./gen_translit_compat.py -u UnicodeData.txt \
	-o $@ --unicode_version $(UNICODE_VERSION)

translit_circle: UnicodeData.txt
translit_circle: gen_translit_circle.py
	$(PYTHON3) ./gen_translit_circle.py -u UnicodeData.txt \
	-o $@ --unicode_version $(UNICODE_VERSION)

translit_cjk_compat: UnicodeData.txt
translit_cjk_compat: gen_translit_cjk_compat.py
	$(PYTHON3) ./gen_translit_cjk_compat.py -u UnicodeData.txt \
	-o $@ --unicode_version $(UNICODE_VERSION)

translit_font: UnicodeData.txt
translit_font: gen_translit_font.py
	$(PYTHON3) ./gen_translit_font.py -u UnicodeData.txt \
	-o $@ --unicode_version $(UNICODE_VERSION)

translit_fraction: UnicodeData.txt
translit_fraction: gen_translit_fraction.py
	$(PYTHON3) ./gen_translit_fraction.py -u UnicodeData.txt \
	-o $@ --unicode_version $(UNICODE_VERSION)

.PHONY: downloads clean-downloads
downloads: $(DOWNLOADS)
clean-downloads:
	-rm -f $(DOWNLOADS)

$(DOWNLOADS):
	$(WGET) http://www.unicode.org/Public/$(UNICODE_VERSION)/ucd/$@
