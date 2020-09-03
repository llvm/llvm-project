# Makefile fragment - requires GNU make
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

S := $(srcdir)/math
B := build/math

math-lib-srcs := $(wildcard $(S)/*.[cS])
math-test-srcs := \
	$(S)/test/mathtest.c \
	$(S)/test/mathbench.c \
	$(S)/test/ulp.c \

math-test-host-srcs := $(wildcard $(S)/test/rtest/*.[cS])

math-includes := $(patsubst $(S)/%,build/%,$(wildcard $(S)/include/*.h))

math-libs := \
	build/lib/libmathlib.so \
	build/lib/libmathlib.a \

math-tools := \
	build/bin/mathtest \
	build/bin/mathbench \
	build/bin/mathbench_libc \
	build/bin/runulp.sh \
	build/bin/ulp \

math-host-tools := \
	build/bin/rtest \

math-lib-objs := $(patsubst $(S)/%,$(B)/%.o,$(basename $(math-lib-srcs)))
math-test-objs := $(patsubst $(S)/%,$(B)/%.o,$(basename $(math-test-srcs)))
math-host-objs := $(patsubst $(S)/%,$(B)/%.o,$(basename $(math-test-host-srcs)))
math-target-objs := $(math-lib-objs) $(math-test-objs)
math-objs := $(math-target-objs) $(math-target-objs:%.o=%.os) $(math-host-objs)

math-files := \
	$(math-objs) \
	$(math-libs) \
	$(math-tools) \
	$(math-host-tools) \
	$(math-includes) \

all-math: $(math-libs) $(math-tools) $(math-includes)

$(math-objs): $(math-includes)
$(math-objs): CFLAGS_ALL += $(math-cflags)
$(B)/test/mathtest.o: CFLAGS_ALL += -fmath-errno
$(math-host-objs): CC = $(HOST_CC)
$(math-host-objs): CFLAGS_ALL = $(HOST_CFLAGS)

$(B)/test/ulp.o: $(S)/test/ulp.h

build/lib/libmathlib.so: $(math-lib-objs:%.o=%.os)
	$(CC) $(CFLAGS_ALL) $(LDFLAGS) -shared -o $@ $^

build/lib/libmathlib.a: $(math-lib-objs)
	rm -f $@
	$(AR) rc $@ $^
	$(RANLIB) $@

$(math-host-tools): HOST_LDLIBS += -lm -lmpfr -lmpc
$(math-tools): LDLIBS += $(math-ldlibs) -lm

build/bin/rtest: $(math-host-objs)
	$(HOST_CC) $(HOST_CFLAGS) $(HOST_LDFLAGS) -o $@ $^ $(HOST_LDLIBS)

build/bin/mathtest: $(B)/test/mathtest.o build/lib/libmathlib.a
	$(CC) $(CFLAGS_ALL) $(LDFLAGS) -static -o $@ $^ $(LDLIBS)

build/bin/mathbench: $(B)/test/mathbench.o build/lib/libmathlib.a
	$(CC) $(CFLAGS_ALL) $(LDFLAGS) -static -o $@ $^ $(LDLIBS)

# This is not ideal, but allows custom symbols in mathbench to get resolved.
build/bin/mathbench_libc: $(B)/test/mathbench.o build/lib/libmathlib.a
	$(CC) $(CFLAGS_ALL) $(LDFLAGS) -static -o $@ $< $(LDLIBS) -lc build/lib/libmathlib.a -lm

build/bin/ulp: $(B)/test/ulp.o build/lib/libmathlib.a
	$(CC) $(CFLAGS_ALL) $(LDFLAGS) -static -o $@ $^ $(LDLIBS)

build/include/%.h: $(S)/include/%.h
	cp $< $@

build/bin/%.sh: $(S)/test/%.sh
	cp $< $@

math-tests := $(wildcard $(S)/test/testcases/directed/*.tst)
math-rtests := $(wildcard $(S)/test/testcases/random/*.tst)

check-math-test: $(math-tools)
	cat $(math-tests) | $(EMULATOR) build/bin/mathtest $(math-testflags)

check-math-rtest: $(math-host-tools) $(math-tools)
	cat $(math-rtests) | build/bin/rtest | $(EMULATOR) build/bin/mathtest $(math-testflags)

check-math-ulp: $(math-tools)
	ULPFLAGS="$(math-ulpflags)" build/bin/runulp.sh $(EMULATOR)

check-math: check-math-test check-math-rtest check-math-ulp

install-math: \
 $(math-libs:build/lib/%=$(DESTDIR)$(libdir)/%) \
 $(math-includes:build/include/%=$(DESTDIR)$(includedir)/%)

clean-math:
	rm -f $(math-files)

.PHONY: all-math check-math-test check-math-rtest check-math-ulp check-math install-math clean-math
