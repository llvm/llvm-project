LEVEL = ../make

C_SOURCES := main-copy.c

include $(LEVEL)/Makefile.rules

# Copy file into the build folder to enable the test to modify it.
main-copy.c: main.c
	cp -f $< $@


clean::
	$(RM) main-copy.c
