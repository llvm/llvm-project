using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    public class DynamicPropertyDescriptor<T> : PropertyDescriptor
    {
        T Value_;
        DynamicPropertyDescriptor<T> Parent_;
        bool IsInheriting_;
        object Component_;

        public DynamicPropertyDescriptor(object Component, DynamicPropertyDescriptor<T> Parent, string Name, Attribute[] Attrs)
            : base(Name, Attrs)
        {
            foreach (DefaultValueAttribute Attr in Attrs.OfType<DefaultValueAttribute>())
            {
                Value_ = (T)Attr.Value;
            }
            Parent_ = Parent;
            IsInheriting_ = true;
            Component_ = Component;
        }

        public bool IsInheriting { get { return IsInheriting_; } set { IsInheriting_ = value; } }
        public DynamicPropertyDescriptor<T> Parent { get { return Parent_; } }

        /// <summary>
        /// Determines whether this property's value should be considered "default" (e.g.
        /// displayed in bold in the property grid).  Root properties are unmodifiable and
        /// always default.  Non-root properties are default iff they are inheriting.
        /// That is to say, if a property is explicitly set to False, the property should
        /// be serialized even if the parent is also False.  It would only not be serialized
        /// if the user had explicitly chosen to inherit it.
        /// </summary>
        /// <param name="component"></param>
        /// <returns></returns>
        public override bool ShouldSerializeValue(object component)
        {
            return (Parent_ != null) && !IsInheriting;
        }

        /// <summary>
        /// Set the value back to the default.  For root properties, this essentially does
        /// nothing as they are read-only anyway.  For non-root properties, this only means
        /// that the property is now inheriting.
        /// </summary>
        /// <param name="component"></param>
        public override void ResetValue(object component)
        {
            IsInheriting_ = true;
        }

        public override void SetValue(object component, object value)
        {
            // This is a bit of a trick.  If the user chose the inheritance option from the
            // dropdown, we will try to set the value to that string.  So look for that and
            // then just reset the value.
            if (value.Equals(MagicInheritance.Text))
                ResetValue(component);
            else
            {
                // By explicitly setting the value, this property is no longer inheriting,
                // even if the value the property is being set to is the same as that of
                // the parent.
                IsInheriting_ = false;
                Value_ = (T)value;
            }
        }

        public override TypeConverter Converter
        {
            get
            {
                // We need to return a DynamicPropertyConverter<> that can deal with our requirement
                // to inject the inherit property option into the dropdown.  But we still need to use
                // the "real" converter to do the actual work for the underlying type.  Therefore,
                // we need to look for a TypeConverter<> attribute on the property, and if it is present
                // forward an instance of that converter to the DynamicPropertyConverter<>.  Otherwise,
                // forward an instance of the default converter for type T to the DynamicPropertyConverter<>.
                TypeConverter UnderlyingConverter = null;
                var ConverterAttr = this.Attributes.OfType<TypeConverterAttribute>().LastOrDefault();
                if (ConverterAttr != null)
                {
                    Type ConverterType = Type.GetType(ConverterAttr.ConverterTypeName);
                    UnderlyingConverter = (TypeConverter)Activator.CreateInstance(ConverterType);
                }
                else
                    UnderlyingConverter = TypeDescriptor.GetConverter(typeof(T));

                return new DynamicPropertyConverter<T>(this, UnderlyingConverter);
            }
        }

        public override bool IsReadOnly
        {
            get
            {
                return (Parent_ == null);
            }
        }

        public override Type ComponentType
        {
            get
            {
                return Component_.GetType();
            }
        }

        public override object GetValue(object component)
        {
            // Return either this property's value or the parents value, depending on
            // whether or not this property is inheriting.
            if (IsInheriting_ && Parent != null)
                return Parent.GetValue(component);
            return Value_;
        }

        public override bool CanResetValue(object component)
        {
            return !IsReadOnly;
        }

        public override Type PropertyType
        {
            get
            {
                return typeof(T);
            }
        }
    }
}
