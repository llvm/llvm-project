# Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#
#	Makefile for iconv data and code.
#
subdir	:= iconvdata

include ../Makeconfig

# Names of all the shared objects which implement the transformations.
modules	:= ISO8859-1 ISO8859-2 ISO8859-3 ISO8859-4 ISO8859-5		 \
	   ISO8859-6 ISO8859-7 ISO8859-8 ISO8859-9 ISO8859-10		 \
	   ISO8859-11 ISO8859-13 ISO8859-14 ISO8859-15 ISO8859-16	 \
	   T.61 ISO_6937 SJIS KOI-8 HP-ROMAN8 HP-ROMAN9 EBCDIC-AT-DE	 \
	   EBCDIC-AT-DE-A EBCDIC-CA-FR EUC-KR UHC JOHAB libJIS libKSC	 \
	   BIG5 EUC-JP libGB EUC-CN libCNS EUC-TW ISO646 EBCDIC-DK-NO	 \
	   EBCDIC-DK-NO-A EBCDIC-ES EBCDIC-ES-A EBCDIC-ES-S EBCDIC-FI-SE \
	   EBCDIC-FI-SE-A EBCDIC-FR EBCDIC-IS-FRISS EBCDIC-IT EBCDIC-PT	 \
	   EBCDIC-UK EBCDIC-US IBM037 IBM038 IBM274 IBM275 IBM423	 \
	   IBM500 IBM870 IBM871 IBM891 IBM903 IBM904 IBM905 IBM1047	 \
	   IBM874 CP737 CP775 ISO-2022-KR HP-TURKISH8 HP-THAI8 HP-GREEK8 \
	   KOI8-R LATIN-GREEK LATIN-GREEK-1 IBM256 IBM273 IBM277 IBM278	 \
	   IBM280 IBM281 IBM284 IBM285 IBM290 IBM297 IBM420 IBM424	 \
	   IBM437 IBM850 IBM851 IBM852 IBM855 IBM857 IBM858 IBM860	 \
	   IBM861 IBM862 IBM863 IBM864 IBM865 IBM868 IBM869 IBM875	 \
	   IBM880 IBM866 CP1258 IBM922 IBM1124 IBM1129 IBM932 IBM943	 \
	   IBM856 IBM930 IBM933 IBM935 IBM937 IBM939 IBM1046		 \
	   IBM1132 IBM1133 IBM1160 IBM1161 IBM1162 IBM1163 IBM1164	 \
	   IBM918 IBM1004 IBM1026 CP1125 CP1250 CP1251 CP1252 CP1253	 \
	   CP1254 CP1255 CP1256 CP1257 ISO-2022-JP MACINTOSH IEC_P27-1	 \
	   ASMO_449 ANSI_X3.110 CSN_369103 CWI DEC-MCS ECMA-CYRILLIC	 \
	   GOST_19768-74 GREEK-CCITT GREEK7 GREEK7-OLD INIS INIS-8	 \
	   INIS-CYRILLIC ISO_6937-2 ISO_2033 ISO_5427 ISO_5427-EXT	 \
	   ISO_5428 ISO_10367-BOX MAC-IS MAC-UK NATS-DANO NATS-SEFI	 \
	   SAMI-WS2 ISO-IR-197 TIS-620 KOI8-U GBK ISIRI-3342 GBGBK	 \
	   ISO-2022-CN libISOIR165 UTF-16 UNICODE UTF-32 UTF-7 BIG5HKSCS \
	   GB18030 ISO-2022-CN-EXT VISCII GBBIG5 CP10007 KOI8-T		 \
	   GEORGIAN-PS GEORGIAN-ACADEMY ISO-IR-209 MAC-SAMI ARMSCII-8	 \
	   TCVN5712-1 libJISX0213 EUC-JISX0213 SHIFT_JISX0213		 \
	   ISO-2022-JP-3 TSCII IBM866NAV CP932 EUC-JP-MS PT154 RK1048	 \
	   IBM1025 IBM1122 IBM1137 IBM1153 IBM1154 IBM1155 IBM1156	 \
	   IBM1157 IBM1158 IBM803 IBM901 IBM902 IBM921 IBM1008		 \
	   IBM1008_420 IBM1097 IBM1112 IBM1123 IBM1130 IBM1140 IBM1141	 \
	   IBM1142 IBM1143 IBM1144 IBM1145 IBM1146 IBM1147 IBM1148	 \
	   IBM1149 IBM1166 IBM1167 IBM4517 IBM4899 IBM4909 IBM4971	 \
	   IBM5347 IBM9030 IBM9066 IBM9448 IBM12712 IBM16804             \
	   IBM1364 IBM1371 IBM1388 IBM1390 IBM1399 ISO_11548-1 MIK BRF	 \
	   MAC-CENTRALEUROPE KOI8-RU ISO8859-9E				 \
	   CP770 CP771 CP772 CP773 CP774

# If lazy binding is disabled, use BIND_NOW for the gconv modules.
ifeq ($(bind-now),yes)
LDFLAGS.so += -Wl,-z,now
endif

modules.so := $(addsuffix .so, $(modules))

ifeq (yes,$(build-shared))
tests = bug-iconv1 bug-iconv2 tst-loading tst-e2big tst-iconv4 bug-iconv4 \
	tst-iconv6 bug-iconv5 bug-iconv6 tst-iconv7 bug-iconv8 bug-iconv9 \
	bug-iconv10 bug-iconv11 bug-iconv12 tst-iconv-big5-hkscs-to-2ucs4 \
	bug-iconv13 bug-iconv14
ifeq ($(have-thread-library),yes)
tests += bug-iconv3
endif

test-srcs := tst-table-from tst-table-to
endif

libJIS-routines := jis0201 jis0208 jis0212
libKSC-routines := ksc5601
libGB-routines := gb2312
libCNS-routines := cns11643l1 cns11643
libISOIR165-routines := iso-ir-165
libJISX0213-routines := jisx0213

LDFLAGS-EUC-KR.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)EUC-KR.so: $(objpfx)libKSC.so
LDFLAGS-JOHAB.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)JOHAB.so: $(objpfx)libKSC.so
LDFLAGS-UHC.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)UHC.so: $(objpfx)libKSC.so
LDFLAGS-EUC-JP.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)EUC-JP.so: $(objpfx)libJIS.so
LDFLAGS-EUC-JP-MS.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)EUC-JP-MS.so: $(objpfx)libJIS.so
LDFLAGS-EUC-CN.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)EUC-CN.so: $(objpfx)libGB.so
LDFLAGS-EUC-TW.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)EUC-TW.so: $(objpfx)libCNS.so
LDFLAGS-EUC-JISX0213.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)EUC-JISX0213.so: $(objpfx)libJISX0213.so
LDFLAGS-SHIFT_JISX0213.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)SHIFT_JISX0213.so: $(objpfx)libJISX0213.so

LDFLAGS-ISO-2022-JP.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)ISO-2022-JP.so: $(objpfx)libJIS.so $(objpfx)libGB.so \
			 $(objpfx)libCNS.so $(objpfx)libKSC.so
LDFLAGS-ISO-2022-JP-3.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)ISO-2022-JP-3.so: $(objpfx)libJIS.so $(objpfx)libJISX0213.so
LDFLAGS-ISO-2022-KR.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)ISO-2022-KR.so: $(objpfx)libKSC.so
LDFLAGS-ISO-2022-CN.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)ISO-2022-CN.so: $(objpfx)libGB.so $(objpfx)libCNS.so
LDFLAGS-ISO-2022-CN-EXT.so = $(LDFLAGS-rpath-ORIGIN)
$(objpfx)ISO-2022-CN-EXT.so: $(objpfx)libGB.so $(objpfx)libCNS.so \
			     $(objpfx)libISOIR165.so

LDFLAGS-libJIS.so = $(LDFLAGS-soname-fname)
LDFLAGS-libKSC.so = $(LDFLAGS-soname-fname)
LDFLAGS-libGB.so = $(LDFLAGS-soname-fname)
LDFLAGS-libCNS.so = $(LDFLAGS-soname-fname)
LDFLAGS-libISOIR165.so = $(LDFLAGS-soname-fname)
LDFLAGS-libJISX0213.so = $(LDFLAGS-soname-fname)

# We build the transformation modules only when we build shared libs.
ifeq (yes,$(build-shared))

# Rule to generate the shared objects.
charmaps = ../localedata/charmaps
-include $(objpfx)iconv-rules
extra-modules-left := $(modules)
include extra-module.mk

gconv-modules = gconv-modules gconv-modules.d/gconv-modules-extra.conf
modpfx = $(objpfx)gconv-modules.d/

extra-objs	+= $(modules.so)
install-others	= $(addprefix $(inst_gconvdir)/, $(modules.so))	\
		  $(addprefix $(inst_gconvdir)/, $(gconv-modules))

# We can build the conversion tables for numerous charsets automatically.

gen-8bit-modules := iso8859-2 iso8859-3 iso8859-4 iso8859-6 iso8859-9 koi-8 \
		    ebcdic-at-de ebcdic-at-de-a ebcdic-ca-fr ebcdic-dk-no   \
		    ebcdic-dk-no-a ebcdic-es-s ebcdic-fi-se ebcdic-fi-se-a  \
		    ebcdic-fr ebcdic-it ebcdic-pt ebcdic-us \
		    ibm037 ibm038 ibm274 ibm275 ibm423 ibm500 ibm870 ibm871 \
		    ibm891 ibm903 ibm904 ibm905 ibm1047

gen-8bit-gap-modules := koi8-r latin-greek latin-greek-1 ibm256 ibm273	   \
			ibm277 ibm278 ibm280 ibm281 ibm284 ibm285 ibm290   \
			ibm297 ibm420 ibm424 ibm437 ibm850 ibm851 ibm852   \
			ibm855 ibm857 ibm858 ibm860 ibm861 ibm862 ibm863   \
			ibm864 ibm865 ibm868 ibm869 ibm875 ibm880 ibm918   \
			ibm1004 ibm1026 cp1125 cp1250 cp1251 cp1252 cp1253 \
			cp1254 cp1256 cp1257 ibm866 iso8859-5 iso8859-7    \
			iso8859-8 iso8859-10 macintosh iec_p27-1 asmo_449  \
			csn_369103 cwi dec-mcs ecma-cyrillic gost_19768-74 \
			greek-ccitt greek7 greek7-old inis inis-8	   \
			inis-cyrillic iso_2033 iso_5427 iso_5427-ext	   \
			iso_5428 iso_10367-box mac-is nats-dano nats-sefi  \
			iso8859-13 iso8859-14 iso8859-15 mac-uk sami-ws2   \
			iso-ir-197 tis-620 koi8-u ibm874 cp10007 koi8-t	   \
			georgian-ps georgian-academy iso-ir-209 mac-sami   \
			iso8859-11 ibm866nav pt154 rk1048 mik brf	   \
			mac-centraleurope koi8-ru hp-roman8 hp-roman9	   \
			ebcdic-es ebcdic-es-a ebcdic-is-friss ebcdic-uk    \
			iso8859-16 viscii iso8859-9e hp-turkish8	   \
			hp-thai8 hp-greek8 cp770 cp771 cp772 cp773 cp774


gen-special-modules := iso8859-7jp

generated-modules := $(gen-8bit-modules) $(gen-8bit-gap-modules) \
		     $(gen-special-modules)

generated += $(generated-modules:=.h) $(generated-modules:=.stmp) \
	     iconv-test.out iconv-rules tst-loading.mtrace	 \
	     mtrace-tst-loading.out tst-tables.out iconv-test.xxx
ifdef objpfx
generated += $(gconv-modules)
endif

# Rules to generate the headers.
export AWK

define generate-8bit-table
$(make-target-directory)
LC_ALL=C $(SHELL) ./gen-8bit.sh $< > $(@:stmp=T)
$(move-if-change) $(@:stmp=T) $(@:stmp=h)
touch $@
endef

define generate-8bit-gap-table
$(make-target-directory)
LC_ALL=C $(SHELL) ./gen-8bit-gap.sh $< > $(@:stmp=T)
$(move-if-change) $(@:stmp=T) $(@:stmp=h)
touch $@
endef

# The headers must be generated before the compilation.
before-compile += $(addprefix $(objpfx),$(generated-modules:=.h))

ifndef avoid-generated
$(objpfx)iconv-rules: Makefile
	$(make-target-directory)
	{ echo $(filter-out lib%, $(modules)); \
	  echo 8bit $(gen-8bit-modules); \
	  echo 8bit-gap $(gen-8bit-gap-modules); } | \
	LC_ALL=C \
	$(AWK) 'NR == 1 { \
		  for (i = 1; i <= NF; i++) { \
		    printf "%s-routines := %s\n", $$i, tolower($$i); \
		    printf "libof-%s := iconvdata\n", tolower($$i); \
		    printf "%s-map := gconv.map\n", $$i; \
		  } \
		}; \
		NR > 1 { \
		  for (i = 2; i <= NF; i++) { \
		    printf "$$(objpfx)%s.stmp: ", $$i; \
		    map = toupper($$i); \
		    sub(/^ISO8859/, "ISO-8859", map); \
		    if (map == "ISO_2033") map = map "-1983"; \
		    printf "$$(charmaps)/%s ", map; \
		    printf "gen-%s.sh\n", $$1; \
		    printf "\t$$(generate-%s-table)\n", $$1; \
		  } \
		}' > $@.new
	mv -f $@.new $@
endif

$(generated-modules:%=$(objpfx)%.h): $(objpfx)%.h: $(objpfx)%.stmp
	@:

$(gen-8bit-modules:%=$(objpfx)%.stmp): $(objpfx)%.stmp: $(objpfx)iconv-rules
$(gen-8bit-gap-modules:%=$(objpfx)%.stmp): $(objpfx)%.stmp: $(objpfx)iconv-rules

$(objpfx)iso8859-7jp.stmp: $(charmaps)/ISO-8859-7 gen-8bit-gap-1.sh
	$(make-target-directory)
	LC_ALL=C $(SHELL) ./gen-8bit-gap-1.sh $< > $(@:stmp=T)
	$(move-if-change) $(@:stmp=T) $(@:stmp=h)
	touch $@

headers: $(addprefix $(objpfx), $(generated-modules:=.h))

$(addprefix $(inst_gconvdir)/, $(modules.so)): \
    $(inst_gconvdir)/%: $(objpfx)% $(+force)
	$(do-install-program)
$(addprefix $(inst_gconvdir)/, $(gconv-modules)): \
    $(inst_gconvdir)/%: $(objpfx)% $(+force)
	$(do-install)
ifeq (no,$(cross-compiling))
# Update the $(prefix)/lib/gconv/gconv-modules.cache file. This is necessary
# if this libc has more gconv modules than the previously installed one.
	if test -f "$(inst_gconvdir)/gconv-modules.cache"; then \
	   LC_ALL=C \
	   $(rtld-prefix) \
	   $(common-objpfx)iconv/iconvconfig \
	     $(addprefix --prefix=,$(install_root)); \
	fi
else
	@echo '*@*@*@ You should recreate $(inst_gconvdir)/gconv-modules.cache'
endif

endif # build-shared = yes

ifeq ($(run-built-tests),yes)
ifeq (yes,$(build-shared))
tests-special += $(objpfx)iconv-test.out $(objpfx)tst-tables.out
ifneq (no,$(PERL))
tests-special += $(objpfx)mtrace-tst-loading.out
endif
endif
endif

ifeq ($(run-built-tests),yes)
LOCALES := \
	de_DE.UTF-8 \
	zh_HK.BIG5-HKSCS \
	$(NULL)

include ../gen-locales.mk

$(objpfx)bug-iconv6.out: $(gen-locales)
$(objpfx)tst-iconv7.out: $(gen-locales)
$(objpfx)tst-iconv-big5-hkscs-to-2ucs4.out: $(gen-locales)
endif

include ../Rules

# Set libof-* for each routine.
cpp-srcs-left := $(modules) $(generated-modules) $(libJIS-routines) \
		 $(libKSC-routines) $(libGB-routines) $(libCNS-routines) \
		 $(libISOIR165-routines) $(libJISX0213-routines)
lib := iconvdata
include $(patsubst %,$(..)libof-iterator.mk,$(cpp-srcs-left))

tst-loading-ENV = MALLOC_TRACE=$(objpfx)tst-loading.mtrace \
		  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)mtrace-tst-loading.out: $(objpfx)tst-loading.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-loading.mtrace > $@; \
	$(evaluate-test)

$(objpfx)bug-iconv1.out: $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
$(objpfx)bug-iconv2.out: $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
$(objpfx)bug-iconv3.out: $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
$(objpfx)bug-iconv5.out: $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
$(objpfx)tst-loading.out: $(addprefix $(objpfx), $(gconv-modules)) \
			  $(addprefix $(objpfx),$(modules.so))
$(objpfx)tst-iconv4.out: $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
$(objpfx)tst-iconv7.out: $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
$(objpfx)bug-iconv10.out: $(addprefix $(objpfx), $(gconv-modules)) \
			  $(addprefix $(objpfx),$(modules.so))
$(objpfx)bug-iconv12.out: $(addprefix $(objpfx), $(gconv-modules)) \
			  $(addprefix $(objpfx),$(modules.so))
$(objpfx)bug-iconv14.out: $(addprefix $(objpfx), $(gconv-modules)) \
			  $(addprefix $(objpfx),$(modules.so))

$(objpfx)iconv-test.out: run-iconv-test.sh \
			 $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so)) \
			 $(common-objdir)/iconv/iconv_prog TESTS
	iconv_modules="$(modules)" \
	$(SHELL) $< $(common-objdir) '$(test-wrapper-env)' \
		 '$(run-program-env)' > $@; \
	$(evaluate-test)

$(objpfx)tst-tables.out: tst-tables.sh \
			 $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so)) \
			 $(objpfx)tst-table-from $(objpfx)tst-table-to
	$(SHELL) $< $(common-objpfx) $(common-objpfx)iconvdata/ \
		'$(test-program-prefix)' > $@; \
	$(evaluate-test)

do-tests-clean common-mostlyclean: tst-tables-clean

.PHONY: tst-tables-clean
tst-tables-clean:
	-rm -f $(objpfx)tst-*.table $(objpfx)tst-EUC-TW.irreversible

$(modpfx):
	mkdir -p $@

$(modpfx)%: % $(modpfx)
	cp $< $@

$(objpfx)gconv-modules: gconv-modules
	cp $^ $@

# Test requires BIG5HKSCS.
$(objpfx)tst-iconv-big5-hkscs-to-2ucs4.out: \
			 $(addprefix $(objpfx), $(gconv-modules)) \
			 $(addprefix $(objpfx),$(modules.so))
