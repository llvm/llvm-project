#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bi04  ########


bi04: run
	

build:  $(SRC)/bi04.f90
	-$(RM) bi04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(SRC)/bi04.f90 -o bi04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bi04.$(OBJX) check.$(OBJX)  $(LIBS) -o bi04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bi04
	bi04.$(EXESUFFIX)

verify: ;

bi04.run: run

