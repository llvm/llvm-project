#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test im20  ########


im20: run
	

build:  $(SRC)/im20.f
	-$(RM) im20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/im20.f -o im20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) im20.$(OBJX) check.$(OBJX) $(LIBS) -o im20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test im20
	im20.$(EXESUFFIX)

verify: ;

im20.run: run

