LEVEL = ../../test/make

CXX_SOURCES := main.cpp

MY_OS = $(shell uname -s)
ifeq "$(MY_OS)" "Darwin"
    LD_EXTRAS ?= -framework LLDB
	FRAMEWORK_INCLUDES=-F/Volumes/data/lldb/svn/trunk/build/Debug
else
    LD_EXTRAS ?= $(LLDB_BUILD_DIR)/_lldb.so
endif

include $(LEVEL)/Makefile.rules
