! The preceding input is compiled by a separate process.
program module_reader
  use pt_mod
  integer, pointer, protected_target :: keep
  type(holder) :: object
  keep => p
  keep => get()
  object = holder(p)
  call read_pt(p)
  call read_value(p)
  nullify(keep)
end
