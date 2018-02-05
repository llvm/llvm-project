LEVEL = ../../make
C_SOURCES := main.c
LD_EXTRAS += -Wl,--build-id=sha1

all: stripped.out

.PHONY: .build-id
stripped.out .build-id: a.out
	$(OBJCOPY) -j .note.gnu.build-id -O binary $< tmp
	rm -rf .build-id
	fn=`od -An -tx1 <tmp|tr -d ' \n'|sed -e 's/^.\{32\}//' -e 's#^..#.build-id/&/#' -e 's#$$#.debug#'` && \
	mkdir -p `dirname $$fn` && \
	$(OBJCOPY) --only-keep-debug $< $$fn && \
	$(OBJCOPY) --strip-all --add-gnu-debuglink=$$fn $< stripped.out
	$(RM) tmp

clean::
	$(RM) -r stripped.out .build-id

include $(LEVEL)/Makefile.rules
