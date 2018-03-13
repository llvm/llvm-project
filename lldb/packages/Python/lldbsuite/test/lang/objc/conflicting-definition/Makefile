LEVEL = ../../../make

LD_EXTRAS = -lobjc -framework Foundation

include $(LEVEL)/Makefile.rules

all: a.out

libTest.dylib:	Test/Test.m
	mkdir -p Test
	$(MAKE) MAKE_DSYM=YES VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/Test/Test.mk all

libTestExt.dylib: TestExt/TestExt.m
	mkdir -p TestExt
	$(MAKE) MAKE_DSYM=YES VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/TestExt/TestExt.mk all

a.out: main.m libTest.dylib libTestExt.dylib
	$(CC) $(LDFLAGS) -I$(SRCDIR) -L. -lTest -lTestExt -o a.out $<

clean::
	rm -rf libTest.dylib libTestExt.dylib a.out Test.o TestExt.o libTest.dylib.dSYM libTestExt.dylib.dSYM
	$(MAKE) MAKE_DSYM=YES VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/Test/Test.mk clean
	$(MAKE) MAKE_DSYM=YES VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/TestExt/TestExt.mk clean
