ifdef INSTALL_ROOT
INSTALL_ROOT_CPPFLAGS = -isystem $(INSTALL_ROOT)/include
INSTALL_ROOT_LDFLAGS = -L$(INSTALL_ROOT)/lib -Wl,-rpath,$(INSTALL_ROOT)/lib
endif

CFLAGS=-Wall -g

LDLIBS = -lpthread

CHECK_SRC := test-1.c test-2.c test-3.c test-6.c test-7.c test-8.c	\
	test-9.c test-10.c test-11.c test-12.c test-13.c test-14.c	\
	test-15.c test-16.c test-17.c test-__pthread_destroy_specific-skip.c

CHECK_OBJS := $(addsuffix .o,$(basename $(notdir $(CHECK_SRC))))
CHECK_PROGS := $(basename $(notdir $(CHECK_SRC))) \
	$(addsuffix -static, $(basename $(CHECK_SRC)))

%.o: %.c
	$(CC) $(INSTALL_ROOT_CPPFLAGS) $(CPPFLAGS) $(CFLAGS) $< -c -o $@

%: %.o
	$(CC) $(INSTALL_ROOT_LDFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

%-static: %.o
	$(CC) -static $(INSTALL_ROOT_LDFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

check: $(CHECK_OBJS) $(CHECK_PROGS)
	for i in $(CHECK_PROGS); do		\
	  echo -n Running $$i...\ ;		\
	  if ./$$i 2>&1 > $$i.out;		\
	  then					\
	    echo Success.;			\
	  else					\
	    echo Failure.;			\
	  fi					\
	done

clean:
	rm -f $(CHECK_OBJS) $(CHECK_PROGS) \
	  $(addsuffix .out,$(basename $(notdir $(CHECK_PROGS))))