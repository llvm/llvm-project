LEVEL = ../../make
C_SOURCES := main.c

all: dirsymlink

dirreal: a.out
	$(RM) -r $@
	mkdir $@
	$(OBJCOPY) --only-keep-debug $< $@/stripped.debug
	$(OBJCOPY) --strip-all --add-gnu-debuglink=$@/stripped.debug $< $@/stripped.out

dirsymlink: dirreal
	$(RM) -r $@
	mkdir $@
	ln -s ../$</stripped.out $@/stripped.symlink

clean::
	$(RM) -r dirreal dirsymlink

include $(LEVEL)/Makefile.rules
