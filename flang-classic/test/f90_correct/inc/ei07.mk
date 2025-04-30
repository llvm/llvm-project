#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei07  ########


ei07: run
	

build:  $(SRC)/ei07.f90
	-$(RM) ei07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei07.f90 -o ei07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei07.$(OBJX) check.$(OBJX) $(LIBS) -o ei07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei07
	ei07.$(EXESUFFIX)

verify: ;

ei07.run: run

