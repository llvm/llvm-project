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

# Makefile for installing locale data source files.

subdir := localedata

include ../Makeconfig

# List with all available character set descriptions.
charmaps := $(wildcard charmaps/[A-I]*) $(wildcard charmaps/[J-Z]*)

# List with all available character set descriptions.
locales := $(wildcard locales/*)


subdir-dirs = tests-mbwc
vpath %.c tests-mbwc
vpath %.h tests-mbwc


test-srcs := collate-test xfrm-test tst-fmon tst-rpmatch tst-trans \
	     tst-ctype tst-langinfo-newlocale tst-langinfo-setlocale \
	     tst-langinfo-newlocale-static tst-langinfo-setlocale-static \
	     tst-numeric
# List of test input files (list sorted alphabetically):
test-input := \
	am_ET.UTF-8 \
	az_AZ.UTF-8 \
	be_BY.UTF-8 \
	ber_DZ.UTF-8 \
	ber_MA.UTF-8 \
	bg_BG.UTF-8 \
	br_FR.UTF-8 \
	bs_BA.UTF-8 \
	ckb_IQ.UTF-8 \
	cmn_TW.UTF-8 \
	crh_UA.UTF-8 \
	cs_CZ.UTF-8 \
	csb_PL.UTF-8 \
	cv_RU.UTF-8 \
	cy_GB.UTF-8 \
	da_DK.ISO-8859-1 \
	de_DE.ISO-8859-1 \
	dsb_DE.UTF-8 \
	dz_BT.UTF-8 \
	en_US.ISO-8859-1 \
	en_US.UTF-8 \
	eo.UTF-8 \
	es_ES.UTF-8 \
	et_EE.UTF-8 \
	fa_IR.UTF-8 \
	fi_FI.UTF-8 \
	fil_PH.UTF-8 \
	fr_CA.UTF-8 \
	fr_FR.UTF-8 \
	fur_IT.UTF-8 \
	gez_ER.UTF-8@abegede \
	ha_NG.UTF-8 \
	hr_HR.ISO-8859-2 \
	hr_HR.UTF-8 \
	hsb_DE.UTF-8 \
	hu_HU.UTF-8 \
	ig_NG.UTF-8 \
	ik_CA.UTF-8 \
	is_IS.UTF-8 \
	kk_KZ.UTF-8 \
	ku_TR.UTF-8 \
	ky_KG.UTF-8 \
	ln_CD.UTF-8 \
	lt_LT.UTF-8 \
	lv_LV.UTF-8 \
	mi_NZ.UTF-8 \
	ml_IN.UTF-8 \
	mn_MN.UTF-8 \
	mr_IN.UTF-8 \
	mt_MT.UTF-8 \
	nan_TW.UTF-8@latin \
	nb_NO.UTF-8 \
	om_KE.UTF-8 \
	or_IN.UTF-8 \
	os_RU.UTF-8 \
	pl_PL.UTF-8 \
	ps_AF.UTF-8 \
	ro_RO.UTF-8 \
	ru_RU.UTF-8 \
	sah_RU.UTF-8 \
	sc_IT.UTF-8 \
	se_NO.UTF-8 \
	si_LK.UTF-8 \
	sq_AL.UTF-8 \
	sr_RS.UTF-8 \
	sv_SE.ISO-8859-1 \
	sv_SE.UTF-8 \
	szl_PL.UTF-8 \
	tg_TJ.UTF-8 \
	tk_TM.UTF-8 \
	tr_TR.UTF-8 \
	tt_RU.UTF-8 \
	tt_RU.UTF-8@iqtelif \
	ug_CN.UTF-8 \
	uk_UA.UTF-8 \
	uz_UZ.UTF-8 \
	vi_VN.UTF-8 \
	yi_US.UTF-8 \
	yo_NG.UTF-8 \
	zh_CN.UTF-8 \
	$(NULL)

test-input-data = $(addsuffix .in, $(test-input))
test-output := $(foreach s, .out .xout, \
			 $(addsuffix $s, $(basename $(test-input))))
ld-test-names := test1 test2 test3 test4 test5 test6 test7
ld-test-srcs := $(addprefix tests/,$(addsuffix .cm,$(ld-test-names)) \
				   $(addsuffix .def,$(ld-test-names)) \
				   $(addsuffix .ds,test5 test6) \
				   test6.c trans.def)

fmon-tests = n01y12 n02n40 n10y31 n11y41 n12y11 n20n32 n30y20 n41n00 \
	     y01y10 y02n22 y22n42 y30y21 y32n31 y40y00 y42n21

generated += $(test-input) $(test-output) sort-test.out tst-locale.out \
	     tst-leaks.mtrace mtrace-tst-leaks.out
generated-dirs += $(ld-test-names) tt_TT de_DE.437			\
		  $(addprefix tstfmon_,$(fmon-tests))			\

ifeq ($(run-built-tests),yes)
locale_test_suite := tst_iswalnum tst_iswalpha tst_iswcntrl            \
		     tst_iswctype tst_iswdigit tst_iswgraph            \
		     tst_iswlower tst_iswprint tst_iswpunct            \
		     tst_iswspace tst_iswupper tst_iswxdigit tst_mblen \
		     tst_mbrlen tst_mbrtowc tst_mbsrtowcs tst_mbstowcs \
		     tst_mbtowc tst_strcoll tst_strfmon tst_strxfrm    \
		     tst_swscanf tst_towctrans tst_towlower            \
		     tst_towupper tst_wcrtomb tst_wcscat tst_wcschr    \
		     tst_wcscmp tst_wcscoll tst_wcscpy tst_wcscspn     \
		     tst_wcslen tst_wcsncat tst_wcsncmp tst_wcsncpy    \
		     tst_wcspbrk tst_wcsrtombs tst_wcsspn tst_wcsstr   \
		     tst_wcstod tst_wcstok tst_wcstombs tst_wcswidth   \
		     tst_wcsxfrm tst_wctob tst_wctomb tst_wctrans      \
		     tst_wctype tst_wcwidth

tests = $(locale_test_suite) tst-digits tst-setlocale bug-iconv-trans \
	tst-leaks tst-mbswcs1 tst-mbswcs2 tst-mbswcs3 tst-mbswcs4 tst-mbswcs5 \
	tst-mbswcs6 tst-xlocale1 tst-xlocale2 bug-usesetlocale \
	tst-strfmon1 tst-sscanf bug-setlocale1 tst-setlocale2 tst-setlocale3 \
	tst-wctype tst-iconv-math-trans
tests-static = bug-setlocale1-static
tests += $(tests-static)
ifeq (yes,$(build-shared))
ifneq (no,$(PERL))
tests-special += $(objpfx)mtrace-tst-leaks.out
endif
endif
endif
tests-container = \
			tst-localedef-hardlinks

# Files to install.
ifeq ($(INSTALL_UNCOMPRESSED),yes)
# This option is for testing inside the testroot container, as the
# container does not include a working gunzip program.
install-others := $(addprefix $(inst_i18ndir)/, \
			      $(charmaps) \
			      $(locales))
else
install-others := $(addprefix $(inst_i18ndir)/, \
			      $(addsuffix .gz, $(charmaps)) \
			      $(locales))
endif

tests: $(objdir)/iconvdata/gconv-modules

tests-static += tst-langinfo-newlocale-static tst-langinfo-setlocale-static

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)sort-test.out $(objpfx)tst-fmon.out \
		 $(objpfx)tst-locale.out $(objpfx)tst-rpmatch.out \
		 $(objpfx)tst-trans.out $(objpfx)tst-ctype.out \
		 $(objpfx)tst-langinfo-newlocale.out \
		 $(objpfx)tst-langinfo-setlocale.out \
		 $(objpfx)tst-langinfo-newlocale-static.out \
		 $(objpfx)tst-langinfo-setlocale-static.out \
		 $(objpfx)tst-numeric.out
# We have to generate locales (list sorted alphabetically)
LOCALES := \
	am_ET.UTF-8 \
	az_AZ.UTF-8 \
	be_BY.UTF-8 \
	ber_DZ.UTF-8 \
	ber_MA.UTF-8 \
	bg_BG.UTF-8 \
	br_FR.UTF-8 \
	bs_BA.UTF-8 \
	ckb_IQ.UTF-8 \
	cmn_TW.UTF-8 \
	crh_UA.UTF-8 \
	cs_CZ.UTF-8 \
	csb_PL.UTF-8 \
	cv_RU.UTF-8 \
	cy_GB.UTF-8 \
	da_DK.ISO-8859-1 \
	de_DE.ISO-8859-1 \
	de_DE.UTF-8 \
	dsb_DE.UTF-8 \
	dz_BT.UTF-8 \
	en_GB.UTF-8 \
	en_US.ANSI_X3.4-1968 \
	en_US.ISO-8859-1\
	en_US.UTF-8 \
	eo.UTF-8 \
	es_ES.UTF-8 \
	et_EE.UTF-8 \
	fa_IR.UTF-8 \
	fi_FI.UTF-8 \
	fil_PH.UTF-8 \
	fr_CA.UTF-8 \
	fr_FR.ISO-8859-1 \
	fr_FR.UTF-8 \
	fur_IT.UTF-8 \
	gez_ER.UTF-8@abegede \
	ha_NG.UTF-8 \
	hr_HR.ISO-8859-2 \
	hr_HR.UTF-8 \
	hsb_DE.UTF-8 \
	hu_HU.UTF-8 \
	ig_NG.UTF-8 \
	ik_CA.UTF-8 \
	is_IS.UTF-8 \
	ja_JP.EUC-JP \
	ja_JP.SJIS \
	ja_JP.UTF-8 \
	kk_KZ.UTF-8 \
	ku_TR.UTF-8 \
	ky_KG.UTF-8 \
	ln_CD.UTF-8 \
	lt_LT.UTF-8 \
	lv_LV.UTF-8 \
	mi_NZ.UTF-8 \
	ml_IN.UTF-8 \
	mn_MN.UTF-8 \
	mr_IN.UTF-8 \
	mt_MT.UTF-8 \
	nan_TW.UTF-8@latin \
	nb_NO.ISO-8859-1 \
	nb_NO.UTF-8 \
	nl_NL.UTF-8 \
	nn_NO.ISO-8859-1 \
	om_KE.UTF-8 \
	or_IN.UTF-8 \
	os_RU.UTF-8 \
	pl_PL.UTF-8 \
	ps_AF.UTF-8 \
	ro_RO.UTF-8 \
	ru_RU.UTF-8 \
	sah_RU.UTF-8 \
	sc_IT.UTF-8 \
	se_NO.UTF-8 \
	si_LK.UTF-8 \
	sq_AL.UTF-8 \
	sr_RS.UTF-8 \
	sv_SE.ISO-8859-1 \
	sv_SE.UTF-8 \
	szl_PL.UTF-8 \
	tg_TJ.UTF-8 \
	tk_TM.UTF-8 \
	tr_TR.ISO-8859-9 \
	tr_TR.UTF-8 \
	tt_RU.UTF-8 \
	tt_RU.UTF-8@iqtelif \
	ug_CN.UTF-8 \
	uk_UA.UTF-8 \
	uz_UZ.UTF-8 \
	vi_VN.UTF-8 \
	yi_US.UTF-8 \
	yo_NG.UTF-8 \
	zh_CN.UTF-8 \
	zh_TW.EUC-TW \
	$(NULL)

include ../gen-locales.mk

$(objpfx)tst-iconv-math-trans.out: $(gen-locales)
endif

include ../Rules

ifeq ($(INSTALL_UNCOMPRESSED),yes)
# Install the charmap files as-is.  This option is for testing inside
# the testroot container, as the container does not include a working
# gunzip program.
$(inst_i18ndir)/charmaps/%: charmaps/% $(+force)
	$(make-target-directory)
	rm -f $@
	$(INSTALL_DATA) $< $@
else
# Install the charmap files in gzipped format.
$(inst_i18ndir)/charmaps/%.gz: charmaps/% $(+force)
	$(make-target-directory)
	rm -f $(@:.gz=) $@
	$(INSTALL_DATA) $< $(@:.gz=)
	gzip -9n $(@:.gz=)
endif

# Install the locale source files in the appropriate directory.
$(inst_i18ndir)/locales/%: locales/% $(+force); $(do-install)

ifeq ($(run-built-tests),yes)
generated-dirs += $(LOCALES)

$(addsuffix .out,$(addprefix $(objpfx),$(tests))): %: \
  $(addprefix $(objpfx),$(CTYPE_FILES))

$(objpfx)sort-test.out: sort-test.sh $(objpfx)collate-test $(objpfx)xfrm-test \
		       $(test-input-data) $(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix-before-env)' \
	  '$(run-program-env)' '$(test-program-prefix-after-env)' \
	  $(test-input) \
	  > $@; \
	$(evaluate-test)
$(objpfx)tst-fmon.out: tst-fmon.sh $(objpfx)tst-fmon tst-fmon.data \
		       $(objpfx)sort-test.out \
		       $(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(run-program-prefix-before-env)' \
		 '$(run-program-env)' '$(run-program-prefix-after-env)' \
		 '$(test-program-prefix)' tst-fmon.data \
	  > $@; \
	$(evaluate-test)
$(objpfx)tst-numeric.out: tst-numeric.sh $(objpfx)tst-numeric tst-numeric.data \
		       $(objpfx)sort-test.out \
		       $(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix)' tst-numeric.data \
	  > $@; \
	$(evaluate-test)
$(objpfx)tst-locale.out: tst-locale.sh $(common-objpfx)locale/localedef \
			 $(ld-test-srcs) $(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(built-program-cmd-before-env)' \
		 '$(run-program-env)' '$(built-program-cmd-after-env)' > $@; \
	$(evaluate-test)
$(objpfx)tst-rpmatch.out: tst-rpmatch.sh $(objpfx)tst-rpmatch \
			  $(objpfx)tst-fmon.out \
			  $(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd)' > $@; \
	$(evaluate-test)
$(objpfx)tst-trans.out: tst-trans.sh $(objpfx)tst-trans \
			$(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(run-program-prefix-before-env)' \
		 '$(run-program-env)' '$(run-program-prefix-after-env)' \
		 '$(test-program-prefix-before-env)' \
		 '$(test-program-prefix-after-env)'; \
	$(evaluate-test)
$(objpfx)tst-ctype.out: tst-ctype.sh $(objpfx)tst-ctype \
			$(objpfx)sort-test.out \
			$(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd-before-env)' \
		 '$(run-program-env)' '$(test-program-cmd-after-env)'; \
	$(evaluate-test)
$(objpfx)tst-langinfo-newlocale.out: tst-langinfo.sh \
			$(objpfx)tst-langinfo-newlocale \
			$(objpfx)sort-test.out \
			$(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd-before-env)' \
		 '$(run-program-env)' '$(test-program-cmd-after-env)' > $@; \
	$(evaluate-test)
$(objpfx)tst-langinfo-newlocale-static.out: tst-langinfo.sh \
			$(objpfx)tst-langinfo-newlocale-static \
			$(objpfx)sort-test.out \
			$(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd-before-env)' \
		 '$(run-program-env)' '$(test-program-cmd-after-env)' > $@; \
	$(evaluate-test)
# Static use of newlocale is known not to work. See Bug 23164.
test-xfail-tst-langinfo-newlocale-static = yes

$(objpfx)tst-langinfo-setlocale.out: tst-langinfo.sh \
			$(objpfx)tst-langinfo-setlocale \
			$(objpfx)sort-test.out \
			$(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd-before-env)' \
		 '$(run-program-env)' '$(test-program-cmd-after-env)' > $@; \
	$(evaluate-test)
$(objpfx)tst-langinfo-setlocale-static.out: tst-langinfo.sh \
			$(objpfx)tst-langinfo-setlocale-static \
			$(objpfx)sort-test.out \
			$(addprefix $(objpfx),$(CTYPE_FILES))
	$(SHELL) $< $(common-objpfx) '$(test-program-cmd-before-env)' \
		 '$(run-program-env)' '$(test-program-cmd-after-env)' > $@; \
	$(evaluate-test)

$(objpfx)tst-digits.out: $(objpfx)tst-locale.out
$(objpfx)tst-mbswcs6.out: $(addprefix $(objpfx),$(CTYPE_FILES))
endif

include SUPPORTED

INSTALL-SUPPORTED-LOCALE-ARCHIVE=$(addprefix install-archive-, $(SUPPORTED-LOCALES))
INSTALL-SUPPORTED-LOCALE-FILES=$(addprefix install-files-, $(SUPPORTED-LOCALES))

# Sometimes the whole collection of locale files should be installed.
LOCALEDEF=I18NPATH=. GCONV_PATH=$(common-objpfx)iconvdata LC_ALL=C \
$(rtld-prefix) $(common-objpfx)locale/localedef
install-locales: install-locale-archive

# Create and install the locale-archive file.
install-locale-archive: $(INSTALL-SUPPORTED-LOCALE-ARCHIVE)
# Create and install the locales individually (no archive).
install-locale-files: $(INSTALL-SUPPORTED-LOCALE-FILES)

install-locales-dir:
	$(..)./scripts/mkinstalldirs $(inst_complocaledir)

# The SHIFT_JIS and SHIFT_JISX0213 character maps are not ASCII compatible,
# therefore we have to use --no-warnings=ascii to disable the ASCII check.
# See localedata/gen-locale.sh for the same logic.
define build-one-locale
	locale=`echo $@ | sed -e 's/^install-[a-z]*-//'`; \
	charset=`echo $$locale | sed -e 's,.*/,,'`; \
	locale=`echo $$locale | sed -e 's,/[^/]*,,'`; \
	if [ "$$charset" = 'SHIFT_JIS' ] \
	   || [ "$$charset" = 'SHIFT_JISX0213' ]; then \
	   flags="$$flags --no-warnings=ascii"; \
	fi; \
	echo -n `echo $$locale | sed 's/\([^.\@]*\).*/\1/'`; \
	echo -n ".$$charset"; \
	echo -n `echo $$locale | sed 's/\([^\@]*\)\(\@.*\)*/\2/'`; \
	echo -n '...'; \
	input=`echo $$locale | sed 's/\([^.]*\)[^@]*\(.*\)/\1\2/'`; \
	$(LOCALEDEF) $$flags --alias-file=../intl/locale.alias \
		     -i locales/$$input -f charmaps/$$charset \
		     $(addprefix --prefix=,$(install_root)) $$locale \
	&& echo ' done';
endef

$(INSTALL-SUPPORTED-LOCALE-ARCHIVE): install-locales-dir
	@flags="-c"; \
	$(build-one-locale)

$(INSTALL-SUPPORTED-LOCALE-FILES): install-locales-dir
	@flags="-c --no-archive --no-hard-links"; \
	$(build-one-locale)

tst-setlocale-ENV = LC_ALL=ja_JP.EUC-JP
tst-wctype-ENV = LC_ALL=ja_JP.EUC-JP

tst-leaks-ENV = MALLOC_TRACE=$(objpfx)tst-leaks.mtrace \
		LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)mtrace-tst-leaks.out: $(objpfx)tst-leaks.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-leaks.mtrace > $@; \
	$(evaluate-test)

bug-setlocale1-ENV-only = LOCPATH=$(objpfx) LC_CTYPE=de_DE.UTF-8
bug-setlocale1-static-ENV-only = $(bug-setlocale1-ENV-only)

$(objdir)/iconvdata/gconv-modules:
	$(MAKE) -C ../iconvdata subdir=iconvdata $@
