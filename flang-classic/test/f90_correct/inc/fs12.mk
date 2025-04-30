#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fs12  ########

# Determine call instruction used
INSN=\(call\|jmp\)
ifeq ($(findstring aarch64, $(UNAME)), aarch64)
    INSN=bl
endif
ifeq ($(findstring ppc64le, $(UNAME)), ppc64le)
    INSN=bl
endif

fs12: run

build:  $(SRC)/fs12.f90
	-$(RM) fs12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fs12.f90 -o fs12.$(OBJX) -Minfo > fs12.txt 2>&1
	-$(FC) $(FFLAGS) $(LDFLAGS) fs12.$(OBJX) check.$(OBJX) $(LIBS) -o fs12.$(EXESUFFIX)

# contig_cpy should not be inlined (except with -Minline=reshape).
# Verify that by checking for exactly 3 calls to f90_mcopy.
# Due to the complexity of counting specific function calls in assembly
# or .ll files, we are now checking -Minfo messages about whether rank2 is
# being inlined.
run:
	@echo ------------------------------------ executing modified test fs12
	@mcopy_calls=`grep -ci 'contig_cpy.*inlined' fs12.txt`; \
	if [ $$mcopy_calls -ne 0 ]; then \
	    echo "RESULT: FAIL" ; \
	    exit 1; \
	else \
	  echo "RESULT: PASS" ; \
	fi
	fs12.$(EXESUFFIX)

verify: ;

fs12.run: run
