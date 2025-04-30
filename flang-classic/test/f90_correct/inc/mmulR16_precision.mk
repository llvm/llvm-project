#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16_precision  ########


mmulR16_precision: run
	

build:  $(SRC)/mmulR16_precision.f08
	-$(RM) mmulR16_precision.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16_precision.f08 -o mmulR16_precision.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16_precision.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR16_precision.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16_precision
	mmulR16_precision.$(EXESUFFIX)

verify: ;

mmulR16_precision.run: run

