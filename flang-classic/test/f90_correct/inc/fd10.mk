#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fd10  ########


fd10: run
	

build:  $(SRC)/fd10.f
	-$(RM) fd10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fd10.f -o fd10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fd10.$(OBJX) check.$(OBJX) $(LIBS) -o fd10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fd10
	fd10.$(EXESUFFIX)

verify: ;

fd10.run: run

