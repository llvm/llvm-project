# defines target $(gen-locales) that generates the locales given in $(LOCALES)

LOCALE_SRCS := $(shell echo "$(LOCALES)"|sed 's/\([^ .]*\)[^@ ]*\(@[^ ]*\)\?/\1\2/g')
CHARMAPS := $(shell echo "$(LOCALES)" | \
		    sed -e 's/[^ .]*[.]\([^@ ]*\)\(@[^@ ]*\)*/\1/g' -e s/SJIS/SHIFT_JIS/g)
CTYPE_FILES = $(addsuffix /LC_CTYPE,$(LOCALES))
gen-locales := $(addprefix $(common-objpfx)localedata/,$(CTYPE_FILES))

# Dependency for the locale files.  We actually make it depend only on
# one of the files.
$(addprefix $(common-objpfx)localedata/,$(CTYPE_FILES)): %: \
  ../localedata/gen-locale.sh \
  $(common-objpfx)locale/localedef \
  ../localedata/Makefile \
  $(addprefix ../localedata/charmaps/,$(CHARMAPS)) \
  $(addprefix ../localedata/locales/,$(LOCALE_SRCS))
	@$(SHELL) ../localedata/gen-locale.sh $(common-objpfx) \
		  '$(built-program-cmd-before-env)' '$(run-program-env)' \
		  '$(built-program-cmd-after-env)' $@; \
	$(evaluate-test)
