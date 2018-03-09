LEVEL = ../../make

include $(LEVEL)/Makefile.rules

LIB_PREFIX := lib

ifeq "$(OS)" "Darwin"
	CFLAGS += -arch $(ARCH)
	DS := dsymutil
	LD_FLAGS := -dynamiclib
	LIB_INDIRECT := $(LIB_PREFIX)indirect.dylib
	LIB_REEXPORT := $(LIB_PREFIX)reexport.dylib
	EXEC_PATH := "@executable_path"
	EXEC_PATH_INDIRECT := -install_name $(EXEC_PATH)/$(LIB_INDIRECT)
	EXEC_PATH_REEXPORT := -install_name $(EXEC_PATH)/$(LIB_REEXPORT)
endif

all: a.out $(LIB_INDIRECT) $(LIB_REEXPORT)

a.out: main.o $(LIB_INDIRECT) $(LIB_REEXPORT)
	$(CC) $(CFLAGS) -o a.out main.o -L. $(LIB_INDIRECT) $(LIB_REEXPORT)

main.o: $(SRCDIR)/main.c
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c

$(LIB_INDIRECT): indirect.o
	$(CC) $(CFLAGS) $(LD_FLAGS) $(EXEC_PATH_INDIRECT) -o $(LIB_INDIRECT) indirect.o
	if [ "$(OS)" = "Darwin" ]; then dsymutil $(LIB_INDIRECT); fi

indirect.o: $(SRCDIR)/indirect.c
	$(CC) $(CFLAGS) -c $(SRCDIR)/indirect.c

$(LIB_REEXPORT): reexport.o $(LIB_INDIRECT)
	$(CC) $(CFLAGS) $(LD_FLAGS) $(EXEC_PATH_REEXPORT) -o $(LIB_REEXPORT) reexport.o -L. -lindirect -Wl,-alias_list,$(SRCDIR)/alias.list
	if [ "$(OS)" = "Darwin" ]; then dsymutil $(LIB_REEXPORT); fi

reexport.o: $(SRCDIR)/reexport.c
	$(CC) $(CFLAGS) -c $(SRCDIR)/reexport.c
clean::
	rm -rf $(wildcard *.o *~ *.dylib *.so a.out *.dSYM)
