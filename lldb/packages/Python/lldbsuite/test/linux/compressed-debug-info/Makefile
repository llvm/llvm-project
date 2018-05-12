LEVEL := ../../make

C_SOURCES := a.c

all: compressed.out compressed.gnu.out

compressed.out: a.out
	$(OBJCOPY) --compress-debug-sections=zlib $< $@

compressed.gnu.out: a.out
	$(OBJCOPY) --compress-debug-sections=zlib-gnu $< $@

include $(LEVEL)/Makefile.rules

clean::
	$(RM) -f a.o main compressed.out compressed.gnu.out
