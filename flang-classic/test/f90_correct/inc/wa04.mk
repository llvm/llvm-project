#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test wa04  ########


wa04: run
	

build:  $(SRC)/wa04.f90
	-$(RM) wa04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/wa04.f90 -o wa04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) wa04.$(OBJX) check.$(OBJX) $(LIBS) -o wa04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test wa04
	wa04.$(EXESUFFIX)

verify: ;

wa04.run: run

