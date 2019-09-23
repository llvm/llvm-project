LEVEL = ../../../make
DYLIB_ONLY := YES
FRAMEWORK := aTestFramework
FRAMEWORK_HEADERS := $(SRCDIR)/aTestFramework.h
FRAMEWORK_MODULES := $(SRCDIR)/module.modulemap
DYLIB_CXX_SOURCES := aTestFramework.mm
LD_EXTRAS := -framework Foundation

include $(LEVEL)/Makefile.rules
