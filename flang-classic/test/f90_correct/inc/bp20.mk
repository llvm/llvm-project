#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bp20  ########


bp20: run
	

build:  $(SRC)/bp20.f
	-$(RM) bp20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bp20.f -o bp20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bp20.$(OBJX) check.$(OBJX) $(LIBS) -o bp20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bp20
	bp20.$(EXESUFFIX)

verify: ;

bp20.run: run

