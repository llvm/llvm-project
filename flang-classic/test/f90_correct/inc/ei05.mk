#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei05  ########


ei05: run
	

build:  $(SRC)/ei05.f90
	-$(RM) ei05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei05.f90 -o ei05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei05.$(OBJX) check.$(OBJX) $(LIBS) -o ei05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei05
	ei05.$(EXESUFFIX)

verify: ;

ei05.run: run

