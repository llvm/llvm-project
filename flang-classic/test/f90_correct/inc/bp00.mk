#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bp00  ########


bp00: run
	

build:  $(SRC)/bp00.f
	-$(RM) bp00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bp00.f -o bp00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bp00.$(OBJX) check.$(OBJX) $(LIBS) -o bp00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bp00
	bp00.$(EXESUFFIX)

verify: ;

bp00.run: run

