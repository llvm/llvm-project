#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls03  ########


ls03: run
	

build:  $(SRC)/ls03.f
	-$(RM) ls03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls03.f -o ls03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls03.$(OBJX) check.$(OBJX) $(LIBS) -o ls03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls03
	ls03.$(EXESUFFIX)

verify: ;

ls03.run: run

