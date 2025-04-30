#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fs11  ########

# Determine call instruction used
INSN=call
OPT=
ifeq ($(findstring aarch64, $(UNAME)), aarch64)
    INSN=bl
endif
ifeq ($(findstring ppc64le, $(UNAME)), ppc64le)
    INSN=bl
endif

fs11: run

build:  $(SRC)/fs11.f90
	-$(RM) fs11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fs11.f90 -o fs11.$(OBJX) -Minfo > fs11.txt 2>&1
	-$(FC) $(FFLAGS) $(LDFLAGS) fs11.$(OBJX) check.$(OBJX) $(LIBS) -o fs11.$(EXESUFFIX)

# rank2 should not be inlined (except with -Minline=reshape).
# Verify that by checking for exactly 3 calls to mmul.
# Due to the complexity of counting specific function calls in assembly
# or .ll files, we are now checking -Minfo messages about whether rank2 is
# being inlined.
run:
	@echo ------------------------------------ executing test fs11
	@mmul_calls=`grep -c 'rank2.*inlined' fs11.txt`; \
	if [ $$mmul_calls -ne 0 ]; then \
	  echo "RESULT: FAIL" ; \
	  exit 1; \
	else \
	  echo "RESULT: PASS" ; \
	fi
	fs11.$(EXESUFFIX)

verify: ;

fs11.run: run
