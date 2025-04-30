#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bi02  ########


bi02: run
	

build:  $(SRC)/bi02.f90
	-$(RM) bi02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(CC) -c $(CFLAGS) $(SRC)/bi02c.c -o bi02_c.$(OBJX)
	-$(FC) -c $(FFLAGS) $(SRC)/bi02.f90 -o bi02_f.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bi02_f.$(OBJX) bi02_c.$(OBJX) check.$(OBJX)  $(LIBS) -o bi02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bi02
	bi02.$(EXESUFFIX)

verify: ;

bi02.run: run

