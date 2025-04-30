ifeq ($(subdir),iconvdata)
override define generate-8bit-table
$(make-target-directory)
LC_ALL=C $(SHELL) ./gen-8bit.sh $< > $(@:stmp=T)
LC_ALL=C $(SHELL) ../sysdeps/s390/multiarch/gen-8bit.sh $< >> $(@:stmp=T)
$(move-if-change) $(@:stmp=T) $(@:stmp=h)
touch $@
endef
endif

ifeq ($(subdir),iconv)
sysdep_routines += gconv_simple
endif
