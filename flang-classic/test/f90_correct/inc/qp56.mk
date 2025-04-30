#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test inttoq  ########


qp56: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp56.f08 fcheck.$(OBJX)
	-$(RM) qp56.$(EXESUFFIX) qp56.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp56.f08 -o qp56.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp56.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp56.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp56
	qp56.$(EXESUFFIX)

verify: ;

qp56.run: run

