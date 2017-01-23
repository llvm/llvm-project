LEVEL = ../../../make

C_SOURCES := test.c


$(info OS: $(OS))
HOST_OS = $(shell uname -s)
ifeq "$(HOST_OS)" "Darwin"
	FRAMEWORK_INCLUDES = -isysroot $(shell xcrun -show-sdk-path)
endif

include $(LEVEL)/Makefile.rules
