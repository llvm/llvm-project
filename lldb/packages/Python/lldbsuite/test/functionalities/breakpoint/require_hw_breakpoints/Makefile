LEVEL = ../../../make

C_SOURCES := main.c

ifneq (,$(findstring icc,$(CC)))
    CFLAGS += -debug inline-debug-info
endif

include $(LEVEL)/Makefile.rules
