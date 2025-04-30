#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do16: do16.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do16.$(OBJX): $(SRC)/do16.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do16.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do16.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do16
run: ;
