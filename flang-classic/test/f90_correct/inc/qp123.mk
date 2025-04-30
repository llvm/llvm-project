#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test stoq  ########


qp123: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp123.f08 fcheck.$(OBJX)
	-$(RM) qp123.$(EXESUFFIX) qp123.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp123.f08 -o qp123.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp123.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp123.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp123
	qp123.$(EXESUFFIX)

verify: ;

qp123.run: run

