#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16_boundary  ########


mmulR16_boundary: run
	

build:  $(SRC)/mmulR16_boundary.f08
	-$(RM) mmulR16_boundary.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16_boundary.f08 -o mmulR16_boundary.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16_boundary.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR16_boundary.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16_boundary
	mmulR16_boundary.$(EXESUFFIX)

verify: ;

mmulR16_boundary.run: run

