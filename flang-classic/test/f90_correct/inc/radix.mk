#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test radix function take quadruple precision  ########


radix: run
	

check.$(OBJX): $(SRC)/check.c
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)

build:  $(SRC)/radix.f08 check.$(OBJX)
	-$(RM) radix.$(EXESUFFIX) radix.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/radix.f08 -o radix.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) radix.$(OBJX) check.$(OBJX) $(LIBS) -o radix.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test radix
	radix.$(EXESUFFIX)

verify: ;

radix.run: run

